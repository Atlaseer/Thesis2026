using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System.Text.Json;

// ---------------------------------------------------------------------------
// Constants — must match StudyPython.py exactly
// ---------------------------------------------------------------------------

class Program
{
    const string TARGET = "compatibility_score";

    // Exactly 9 base features. No engineered features are added in either
    // language so both models train on identical inputs.
    static readonly string[] BASE_FEATURES =
    {
        "skill_match_score",
        "skill_complementarity_score",
        "network_value_a_to_b",
        "network_value_b_to_a",
        "career_alignment_score",
        "experience_gap",
        "industry_match",
        "geographic_score",
        "seniority_match",
    };

    // Dataset-size fractions — must match StudyPython.py
    static readonly double[] PERCENTAGES = { 1.00, 0.75, 0.50, 0.25, 0.10 };

    // ---------------------------------------------------------------------------
    // Timing helper
    // ---------------------------------------------------------------------------

    static double NowNs() =>
        Stopwatch.GetTimestamp() * 1_000_000_000.0 / Stopwatch.Frequency;

    // ---------------------------------------------------------------------------
    // Entry point
    // ---------------------------------------------------------------------------

    static void Main(string[] args)
    {
        string csvPath = @"C:\Code\Thesis\data\compatibility_pairs.csv";
        int    seed     = 42;
        double testSize = 0.2;
        int    repeats  = 100;
        int    warmup   = 1;

        if (!File.Exists(csvPath))
            throw new FileNotFoundException($"CSV not found: {csvPath}");

        // Load and clean once — avoids mixing disk/cache effects with repeat timing.
        Console.WriteLine("Loading and preprocessing data...");
        var (data, features, loadNs, cleanNs) = LoadCleanAndSelect(csvPath);
        Console.WriteLine($"Preprocessing done. Rows after cleaning: {data.Count}, " +
                          $"features used: [{string.Join(", ", features)}]");

        // Build subset sizes from the shared percentage list.
        var sizes = PERCENTAGES.Select(p => Math.Max(1, (int)(data.Count * p))).ToArray();

        string outDir  = "results";
        string outPath = Path.Combine(outDir, "csharp_timings.csv");
        Directory.CreateDirectory(outDir);

        var results = new List<ResultRow>();
        var modelTypes = new[] { "linear", "tree" };

        foreach (var n in sizes)
        {
            int nEff = Math.Min(n, data.Count);

            // Simple random sample with fixed seed — identical logic to Python
            // df.sample(n=nEff, random_state=seed).
            var rndSample = new Random(seed);
            var subset = data
                .OrderBy(_ => rndSample.Next())
                .Take(nEff)
                .ToList();

            foreach (var modelType in modelTypes)
            {
                Console.WriteLine($"  subset={nEff,7}  model={modelType}");

                // Warmup runs — not recorded.
                for (int w = 0; w < warmup; w++)
                    SplitTrainInfer(subset, features, testSize, seed, modelType);

                // Measured repeats.
                for (int r = 0; r < repeats; r++)
                {
                    var times = SplitTrainInfer(subset, features, testSize, seed, modelType);

                    // preprocess_ns and total_ns defined identically to StudyPython.py:
                    //   preprocess_ns = load_ns + clean_ns + split_ns
                    //   total_ns      = load_ns + clean_ns + split_ns + train_ns + infer_ns
                    double preprocessNs = loadNs + cleanNs + times.SplitNs;
                    double totalNs      = preprocessNs + times.TrainNs + times.InferNs;

                    results.Add(new ResultRow
                    {
                        Language     = "csharp",
                        Library      = "ml.net",
                        Model        = modelType,
                        SubsetSize   = nEff,
                        Repeat       = r,
                        NFeatures    = features.Count,
                        Seed         = seed,
                        SplitSeed    = seed,
                        TestSize     = testSize,
                        LoadNs       = loadNs,
                        CleanNs      = cleanNs,
                        SplitNs      = times.SplitNs,
                        PreprocessNs = preprocessNs,
                        TrainNs      = times.TrainNs,
                        InferNs      = times.InferNs,
                        TotalNs      = totalNs,
                    });
                }
            }
        }

        // Write CSV — column order matches StudyPython.py output exactly.
        using (var writer = new StreamWriter(outPath))
        {
            writer.WriteLine(
                "language,library,model,subset_size,repeat,n_features," +
                "seed,split_seed,test_size," +
                "load_ns,clean_ns,split_ns,preprocess_ns,train_ns,infer_ns,total_ns");

            foreach (var row in results)
            {
                writer.WriteLine(
                    $"{row.Language},{row.Library},{row.Model}," +
                    $"{row.SubsetSize},{row.Repeat},{row.NFeatures}," +
                    $"{row.Seed},{row.SplitSeed},{row.TestSize}," +
                    $"{row.LoadNs},{row.CleanNs},{row.SplitNs},{row.PreprocessNs}," +
                    $"{row.TrainNs},{row.InferNs},{row.TotalNs}");
            }
        }

        // Write meta JSON.
        var meta = new
        {
            csv                 = Path.GetFullPath(csvPath),
            percentages         = PERCENTAGES,
            sizes_used          = sizes,
            repeats,
            warmup,
            seed,
            test_size           = testSize,
            target              = TARGET,
            base_features       = BASE_FEATURES,
            features_used       = features,
            rows_after_cleaning = data.Count,
            notes = new
            {
                linear_model    = "OlsRegression — closed-form exact solver (no iterations). Matches Python LinearRegression.",
                tree_model      = "FastForest(numberOfTrees:1) — single unpruned regression tree. Matches Python DecisionTreeRegressor.",
                precision       = "All feature/target arrays are double (float64). Matches Python float64 arrays.",
                preprocess_ns   = "load_ns + clean_ns + split_ns",
                total_ns        = "load_ns + clean_ns + split_ns + train_ns + infer_ns",
                load_clean_once = "CSV is loaded and cleaned once to avoid mixing disk/cache effects with repeat timing.",
                no_feature_eng  = "No engineered features added. Both languages use exactly the same BASE_FEATURES columns.",
            }
        };

        File.WriteAllText(
            Path.ChangeExtension(outPath, ".meta.json"),
            JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true })
        );

        Console.WriteLine($"Wrote: {Path.GetFullPath(outPath)}");
        Console.WriteLine($"Wrote: {Path.ChangeExtension(Path.GetFullPath(outPath), ".meta.json")}");
    }

    // ---------------------------------------------------------------------------
    // Load, clean, and select features
    // ---------------------------------------------------------------------------

    static (List<Dictionary<string, double>>, List<string>, double, double)
        LoadCleanAndSelect(string path)
    {
        // --- Load phase ---
        double t0 = NowNs();

        List<Dictionary<string, double>> rows;

        using (var reader = new StreamReader(path))
        using (var csv    = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            var records = csv.GetRecords<dynamic>().ToList();

            // 1. Numeric coercion: parse only the model columns, store as double
            //    (float64) to match Python. Columns that fail to parse are omitted
            //    from the dict (caught by the NaN-filter step below).
            rows = records.Select(r =>
            {
                var raw  = (IDictionary<string, object>)r;
                var dict = new Dictionary<string, double>();

                foreach (var f in BASE_FEATURES.Append(TARGET))
                {
                    if (!raw.TryGetValue(f, out var val)) continue;
                    var str = val?.ToString()?.Trim();
                    if (double.TryParse(str, NumberStyles.Any,
                                        CultureInfo.InvariantCulture, out double num))
                        dict[f] = num;
                }

                return dict;
            }).ToList();
        }

        double loadNs = NowNs() - t0;

        // --- Clean phase ---
        double t1 = NowNs();

        var allCols = BASE_FEATURES.Append(TARGET).ToArray();

        // 2. Drop rows where any model column is missing, NaN, or Infinity.
        //    Matches Python: valid_mask = notna().all() & isfinite().all()
        rows = rows
            .Where(r => allCols.All(f =>
                r.ContainsKey(f) &&
                !double.IsNaN(r[f]) &&
                !double.IsInfinity(r[f])))
            .ToList();

        // 3. Drop exact duplicate rows (keep first occurrence).
        //    Key = all model column values joined — matches Python drop_duplicates().
        rows = rows
            .GroupBy(r => string.Join("|", allCols.Select(f => r[f])))
            .Select(g => g.First())
            .ToList();

        // 4. Drop constant features (nunique <= 1) — matches Python detect_constant_cols().
        var features = BASE_FEATURES
            .Where(f => rows.Select(r => r[f]).Distinct().Count() > 1)
            .ToList();

        double cleanNs = NowNs() - t1;

        return (rows, features, loadNs, cleanNs);
    }

    // ---------------------------------------------------------------------------
    // Per-repeat timing: split → train → infer
    // ---------------------------------------------------------------------------

    static (double SplitNs, double TrainNs, double InferNs)
        SplitTrainInfer(
            List<Dictionary<string, double>> data,
            List<string> features,
            double testSize,
            int seed,
            string modelType)
    {
        // --- Split phase ---
        // Shuffle with fixed seed then slice — equivalent to Python train_test_split
        // with random_state=seed and test_size=testSize.
        double t0 = NowNs();

        var rnd      = new Random(seed);
        var shuffled = data.OrderBy(_ => rnd.Next()).ToList();
        int splitIdx = (int)(shuffled.Count * (1.0 - testSize));
        var train    = shuffled.Take(splitIdx).ToList();
        var test     = shuffled.Skip(splitIdx).ToList();

        double splitNs = NowNs() - t0;

        // Build ML.NET IDataView from double[] feature vectors.
        var ml         = new MLContext(seed: seed);
        int nFeatures  = features.Count;

        var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
        schemaDef[nameof(ModelInput.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Double, nFeatures);

        var trainData = train.Select(r => new ModelInput
        {
            Features = features.Select(f => r[f]).ToArray(),
            Label    = r[TARGET],
        }).ToList();

        var testData = test.Select(r => new ModelInput
        {
            Features = features.Select(f => r[f]).ToArray(),
            Label    = r[TARGET],
        }).ToList();

        var trainView = ml.Data.LoadFromEnumerable(trainData, schemaDef);
        var testView  = ml.Data.LoadFromEnumerable(testData,  schemaDef);

        // --- Train phase ---
        // linear → OlsRegression  (closed-form exact solver, matches Python LinearRegression)
        // tree   → FastForest(numberOfTrees:1)  (single regression tree, matches Python
        //          DecisionTreeRegressor — no bagging, no ensemble averaging)
        double t1;
        ITransformer model;

        if (modelType == "linear")
        {
            var pipeline = ml.Regression.Trainers.Ols(
                labelColumnName:   "Label",
                featureColumnName: "Features");

            t1    = NowNs();
            model = pipeline.Fit(trainView);
        }
        else if (modelType == "tree")
        {
            var pipeline = ml.Regression.Trainers.FastForest(
                labelColumnName:   "Label",
                featureColumnName: "Features",
                numberOfTrees:     1,          // single tree — no ensemble
                numberOfLeaves:    checked((int)Math.Pow(2, 20)), // uncapped depth proxy
                minimumExampleCountPerLeaf: 1  // allow pure leaves — matches sklearn default
            );

            t1    = NowNs();
            model = pipeline.Fit(trainView);
        }
        else
        {
            throw new ArgumentException($"Unknown modelType: {modelType}");
        }

        double trainNs = NowNs() - t1;

        // --- Inference phase ---
        double t2    = NowNs();
        model.Transform(testView);
        double inferNs = NowNs() - t2;

        return (splitNs, trainNs, inferNs);
    }

    // ---------------------------------------------------------------------------
    // Data classes
    // ---------------------------------------------------------------------------

    // Feature vector uses double[] to match Python float64 arrays.
    class ModelInput
    {
        [VectorType]
        public double[] Features { get; set; } = default!;
        public double Label { get; set; }
    }

    class ResultRow
    {
        public string Language     = "";
        public string Library      = "";
        public string Model        = "";
        public int    SubsetSize;
        public int    Repeat;
        public int    NFeatures;
        public int    Seed;
        public int    SplitSeed;
        public double TestSize;
        public double LoadNs;
        public double CleanNs;
        public double SplitNs;
        public double PreprocessNs;
        public double TrainNs;
        public double InferNs;
        public double TotalNs;
    }
}
