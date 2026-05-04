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
using Microsoft.ML.Trainers;
using System.Text.Json;

// ----------------------------------------
// Constants are matching the Python code
// ----------------------------------------

class Program
{
    const string TARGET = "compatibility_score";

    // ---------------------------------------------------------------------------------
    // Uses 9 base features, no engineered features
    // ---------------------------------------------------------------------------------

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
        "seniority_match"
    };

    // ---------------------------------------------------------------------------------
    // Set dataset fractions to test
    // ---------------------------------------------------------------------------------
    static readonly double[] PERCENTAGES = { 1.00, 0.75, 0.50, 0.25, 0.10 };

    //static readonly double[] PERCENTAGES = { 1.00, 0.75, 0.50, 0.25, 0.10 };

    // -------------------------------------------------------------------------
    // Timing: call NowNs() before and after a block, then subtract.
    // BUG FIX: the old LoadCleanAndSelect stored a raw Stopwatch.GetTimestamp()
    // value instead of a duration (~2e16 ns = 231 days). Fixed by always
    // computing (NowNs() - t0) immediately after the timed block.
    // -------------------------------------------------------------------------
    static double NowNs() =>
        Stopwatch.GetTimestamp() * 1_000_000_000.0 / Stopwatch.Frequency;

    // -------------------------------------------------------------------------
    // OS physical RAM for this process. proc.Refresh() is mandatory —
    // WorkingSet64 caches stale values without it.
    // Comparable to psutil RSS in Python: captures managed heap + native buffers.
    // -------------------------------------------------------------------------
    static long GetRssBytes()
    {
        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        return proc.WorkingSet64;
    }

    // ---------------------------------------------------------------------------------
    // Entry point which runs the experiment: load, clean, select features, 
    // then for each subset size: 
    // warmup, then repeat: split, train, infer, and record timings.
    // ---------------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // .NET GC managed heap only. Undercounts ML.NET native allocations
    // (e.g. FastForest tree buffers live outside the GC heap).
    // -------------------------------------------------------------------------
    static long GetHeapBytes() => GC.GetTotalMemory(false);

    static void Main(string[] args)
    {
        string csvPath = @"C:\Code\Thesis\data\compatibility_pairs.csv";
        int seed = 42;
        float testSize = 0.2f;
        int repeats = 5;
        int warmup = 0;

        if (!File.Exists(csvPath))
            throw new FileNotFoundException($"CSV not found: {csvPath}");

        Console.WriteLine("Loading and preprocessing data...");
        var (data, features,
             loadNs, loadHeapDelta, loadRssDelta,
             cleanNs, cleanHeapDelta, cleanRssDelta)
            = LoadCleanAndSelect(csvPath);

        Console.WriteLine($"Done. Rows after cleaning: {data.Count}, features: [{string.Join(", ", features)}]");
        Console.WriteLine($"  load time: {loadNs / 1e6:F2} ms | heap {loadHeapDelta / 1024:+#;-#;0} KB | RSS {loadRssDelta / 1024:+#;-#;0} KB");
        Console.WriteLine($"  clean time: {cleanNs / 1e6:F2} ms | heap {cleanHeapDelta / 1024:+#;-#;0} KB | RSS {cleanRssDelta / 1024:+#;-#;0} KB");

        var sizes = PERCENTAGES.Select(p => Math.Max(1, (int)(data.Count * p))).ToArray();

        string outDir = "results";
        string outPath = Path.Combine(outDir, "csharp_timings.csv");
        Directory.CreateDirectory(outDir);

        var results = new List<ResultRow>();
        var modelTypes = new[] { "linear", "tree" };

        foreach (var n in sizes)
        {
            int nEff = Math.Min(n, data.Count);
            var rndSample = new Random(seed);
            var subset = data.OrderBy(_ => rndSample.Next()).Take(nEff).ToList();

            Console.WriteLine($"\nSubset size: {nEff}");
            foreach (var modelType in modelTypes)
            {
                Console.WriteLine($"  model={modelType}");

                // Warmup — not recorded
                for (int w = 0; w < warmup; w++)
                    SplitTrainInfer(subset, features, testSize, seed, modelType);

                for (int r = 0; r < repeats; r++)
                {
                    var t = SplitTrainInfer(subset, features, testSize, seed, modelType);

                    // preprocess_ns = load + clean + split (mirrors Python definition)
                    double preprocessNs = loadNs + cleanNs + t.SplitNs;
                    double pipelineNs   = t.SplitNs + t.TrainNs + t.InferNs;  // repeatable phases only
                    double totalNs      = preprocessNs + t.TrainNs + t.InferNs; // kept for reference

                    results.Add(new ResultRow
                    {
                        Language = "csharp",
                        Library = "ml.net",
                        Model = modelType,
                        SubsetSize = nEff,
                        Repeat = r,
                        NFeatures = features.Count,
                        Seed = seed,
                        SplitSeed = seed,
                        TestSize = testSize,
                        LoadNs = loadNs,
                        LoadHeapDeltaBytes = loadHeapDelta,
                        LoadRssDeltaBytes = loadRssDelta,
                        CleanNs = cleanNs,
                        CleanHeapDeltaBytes = cleanHeapDelta,
                        CleanRssDeltaBytes = cleanRssDelta,
                        SplitNs = t.SplitNs,
                        SplitHeapDeltaBytes = t.SplitHeapDelta,
                        SplitRssDeltaBytes = t.SplitRssDelta,
                        PreprocessNs = preprocessNs,
                        PipelineNs = pipelineNs,
                        TrainNs = t.TrainNs,
                        TrainHeapDeltaBytes = t.TrainHeapDelta,
                        TrainRssDeltaBytes = t.TrainRssDelta,
                        InferNs = t.InferNs,
                        InferHeapDeltaBytes = t.InferHeapDelta,
                        InferRssDeltaBytes = t.InferRssDelta,
                        TotalNs = totalNs,
                        R2 = t.R2,
                        Rmse = t.Rmse,
                    });
                }
            }
        }

        // BUG FIX: header and writer now include all memory columns.
        // BUG FIX: test_size uses :F2 so 0.2f serialises as "0.20" not "0".
        using (var writer = new StreamWriter(outPath))
        {
            writer.WriteLine(
                "language,library,model,subset_size,repeat,n_features,seed,split_seed,test_size," +
                "load_ns,load_heap_peak_bytes,load_rss_delta_bytes," +
                "clean_ns,clean_heap_peak_bytes,clean_rss_delta_bytes," +
                "split_ns,split_heap_peak_bytes,split_rss_delta_bytes," +
                "preprocess_ns," +
                "pipeline_ns," +
                "train_ns,train_heap_peak_bytes,train_rss_delta_bytes," +
                "infer_ns,infer_heap_peak_bytes,infer_rss_delta_bytes," +
                "total_ns," +
                "r2," +
                "rmse," +
                "stratify_by_target,derive_network_asymmetry,vary_split_per_repeat");

            foreach (var row in results)
            {
                var ci = CultureInfo.InvariantCulture;
                writer.WriteLine(
                    string.Join(",", new object[]
                    {
                        row.Language,
                        row.Library,
                        row.Model,
                        row.SubsetSize,
                        row.Repeat,
                        row.NFeatures,
                        row.Seed,
                        row.SplitSeed,
                        row.TestSize.ToString("F2", ci),
                        row.LoadNs.ToString(ci),
                        row.LoadHeapDeltaBytes.ToString(ci),
                        row.LoadRssDeltaBytes.ToString(ci),
                        row.CleanNs.ToString(ci),
                        row.CleanHeapDeltaBytes.ToString(ci),
                        row.CleanRssDeltaBytes.ToString(ci),
                        row.SplitNs.ToString(ci),
                        row.SplitHeapDeltaBytes.ToString(ci),
                        row.SplitRssDeltaBytes.ToString(ci),
                        row.PreprocessNs.ToString(ci),
                        row.PipelineNs.ToString(ci),
                        row.TrainNs.ToString(ci),
                        row.TrainHeapDeltaBytes.ToString(ci),
                        row.TrainRssDeltaBytes.ToString(ci),
                        row.InferNs.ToString(ci),
                        row.InferHeapDeltaBytes.ToString(ci),
                        row.InferRssDeltaBytes.ToString(ci),
                        row.TotalNs.ToString(ci),
                        row.R2.ToString("F6", ci),
                        row.Rmse.ToString("F6", ci),
                        row.StratifyByTarget,
                        row.DeriveNetworkAsymmetry,
                        row.VarySpiltPerRepeat,
                    })
                );
            }
        }

        var meta = new
        {
            csv = Path.GetFullPath(csvPath),
            percentages = PERCENTAGES,
            sizes_used = sizes,
            repeats,
            warmup,
            seed,
            test_size = testSize,
            target = TARGET,
            base_features = BASE_FEATURES,
            features_used = features,
            rows_after_cleaning = data.Count,
            notes = new
            {
                precision_mismatch = "C# feature arrays cast to float32 for ML.NET compatibility. Python uses float64. May affect tree model timing slightly.",
                linear_model = "OlsRegression — closed-form exact solver. Matches Python LinearRegression.",
                tree_model = "FastForest(numberOfTrees:1) — single unpruned tree. Matches Python DecisionTreeRegressor.",
                preprocess_ns = "load_ns + clean_ns + split_ns",
                total_ns = "load_ns + clean_ns + split_ns + train_ns + infer_ns",
                split_ns = "Shuffle + partition + float32 conversion + ML.NET IDataView materialisation per repeat. Equivalent to Python's to_numpy() + train_test_split().",
                load_clean_once = "Load and clean are timed once; their ns/heap/rss values are identical across all repeats.",
                heap_delta_bytes = "GC.GetTotalMemory delta. Managed heap only — undercounts ML.NET native buffers (e.g. FastForest tree storage).",
                rss_delta_bytes = "Peak WorkingSet64 polled every 5 ms during the phase (PeakRssBytesDuring). Captures native heap peaks (e.g. FastForest buffers) that a point-in-time snapshot misses. Comparable to Python psutil RSS.",
                bug_fix_load_ns = "Previously load_ns stored a raw Stopwatch.GetTimestamp() value (~2e16). Now correctly stores elapsed ns.",
                bug_fix_test_size = "Previously test_size serialised as '0' (missing format specifier). Now uses :F2.",
                bug_fix_csv_columns = "Previously memory columns existed in ResultRow but were absent from the CSV header and writer rows.",
                r2 = "R² (coefficient of determination) from ml.Regression.Evaluate(), computed on the test set after inference, outside timed blocks.",
                rmse = "Root Mean Squared Error on the test set, computed outside timed blocks.",
            }
        };

        File.WriteAllText(
            Path.ChangeExtension(outPath, ".meta.json"),
            JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"\nWrote: {Path.GetFullPath(outPath)}");
        Console.WriteLine($"Wrote: {Path.ChangeExtension(Path.GetFullPath(outPath), ".meta.json")}");
    }

    // -------------------------------------------------------------------------
    // Load CSV and clean data. Measures time + memory for each phase separately.
    // -------------------------------------------------------------------------
    static (List<Dictionary<string, double>> data,
        List<string> features,
        double loadNs, long loadHeapDelta, long loadRssDelta,
        double cleanNs, long cleanHeapDelta, long cleanRssDelta)
        LoadCleanAndSelect(string path)
    {
    // --- Load phase ---
    long heapBefore = GetHeapBytes();
    long rssBefore  = GetRssBytes();
    double t0       = NowNs();

    List<Dictionary<string, string>> rawRows;
    using (var reader = new StreamReader(path))
    using (var csv    = new CsvReader(reader, CultureInfo.InvariantCulture))
    {
        var records = csv.GetRecords<dynamic>().ToList();
        rawRows = records.Select(r =>
        {
            var raw = (IDictionary<string, object>)r;
            // Store every column as a raw string — no parsing, no filtering
            return raw.ToDictionary(kv => kv.Key, kv => kv.Value?.ToString()?.Trim() ?? "");
        }).ToList();
    }

    double loadNs      = NowNs() - t0;
    long loadHeapDelta = GetHeapBytes() - heapBefore;
    long loadRssDelta  = GetRssBytes()  - rssBefore;

    // --- Clean phase ---
    long   heapBefore2 = GetHeapBytes();
    long   rssBefore2  = GetRssBytes();
    double t1          = NowNs();

    var allCols = BASE_FEATURES.Append(TARGET).ToArray();

    // Dedup on ALL columns first — mirrors Python's df.drop_duplicates() on the full DataFrame
    rawRows = rawRows
        .GroupBy(r => string.Join("|", r.OrderBy(kv => kv.Key).Select(kv => kv.Value)))
        .Select(g => g.First())
        .ToList();

    // Parse model columns only, dropping rows where any field fails to parse,
    // is NaN, or is Infinity — mirrors Python's to_numeric(errors="coerce") + notna()
    var rows = rawRows
        .Select(r =>
        {
            var dict = new Dictionary<string, double>();
            foreach (var f in allCols)
            {
                if (!r.TryGetValue(f, out var str)
                    || !double.TryParse(str, NumberStyles.Any,
                                        CultureInfo.InvariantCulture, out double num)
                    || double.IsNaN(num)
                    || double.IsInfinity(num))
                    return null; // coerce failure → drop row
                dict[f] = num;
            }
            return dict;
        })
        .Where(r => r != null)
        .Select(r => r!)
        .ToList();

    // Drop constant features (nunique <= 1) — same as Python's detect_constant_numeric_cols
    var features = BASE_FEATURES
        .Where(f => rows.Select(r => r[f]).Distinct().Count() > 1)
        .ToList();

    double cleanNs      = NowNs() - t1;
    long cleanHeapDelta = GetHeapBytes() - heapBefore2;
    long cleanRssDelta  = GetRssBytes()  - rssBefore2;

    return (rows, features,
            loadNs,  loadHeapDelta,  loadRssDelta,
            cleanNs, cleanHeapDelta, cleanRssDelta);
}

    // -------------------------------------------------------------------------
    // Return type carrying timing + memory for all three phases.
    // -------------------------------------------------------------------------
    record PhaseResult(
        double SplitNs, long SplitHeapDelta, long SplitRssDelta,
        double TrainNs, long TrainHeapDelta, long TrainRssDelta,
        double InferNs, long InferHeapDelta, long InferRssDelta,
        double R2, double Rmse);

    // -------------------------------------------------------------------------
    // Split → Train → Infer. Each phase is timed and memory-measured
    // independently so per-phase costs can be compared against Python.
    // -------------------------------------------------------------------------
    static PhaseResult SplitTrainInfer(
        List<Dictionary<string, double>> data,
        List<string> features,
        float testSize,
        int seed,
        string modelType)
    {
        var featureCount = features.Count;
        var ml = new MLContext(seed: seed);

        // --- Split phase: shuffle, partition, convert to float32, load into ML.NET ---
        long splitHeapBefore = GetHeapBytes();
        double t0 = NowNs();

        IDataView trainCached = null!;
        IDataView testCached  = null!;
        long splitRssPeak = PeakRssBytesDuring(() =>
        {
            var rnd = new Random(seed);
            var shuffled = data.OrderBy(_ => rnd.Next()).ToList();
            int splitIdx = (int)(shuffled.Count * (1 - testSize));
            var trainList = shuffled.Take(splitIdx).ToList();
            var testList  = shuffled.Skip(splitIdx).ToList();

            var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
            schemaDef[nameof(ModelInput.Features)].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, featureCount);

            var trainView = ml.Data.LoadFromEnumerable(
                trainList.Select(r => new ModelInput {
                    Features = features.Select(f => (float)r[f]).ToArray(),
                    Label = (float)r[TARGET]
                }), schemaDef);
            var testView = ml.Data.LoadFromEnumerable(
                testList.Select(r => new ModelInput {
                    Features = features.Select(f => (float)r[f]).ToArray(),
                    Label = (float)r[TARGET]
                }), schemaDef);

            // Force materialisation — equivalent to Python's to_numpy() copy
            trainCached = ml.Data.Cache(trainView);
            testCached  = ml.Data.Cache(testView);
            using (var cur = trainCached.GetRowCursor(trainCached.Schema))
                while (cur.MoveNext()) { }
            using (var cur = testCached.GetRowCursor(testCached.Schema))
                while (cur.MoveNext()) { }
        });

        double splitNs      = NowNs() - t0;
        long splitHeapDelta = GetHeapBytes() - splitHeapBefore;
        long splitRssDelta  = splitRssPeak;


        // --- Train phase ---
        long trainHeapBefore = GetHeapBytes();
        double t1 = NowNs();

        ITransformer model = null!;
        long trainRssPeak = PeakRssBytesDuring(() =>
        {
            if (modelType == "linear")
            {
                model = ml.Regression.Trainers.Ols(
                    labelColumnName: "Label",
                    featureColumnName: "Features")
                    .Fit(trainCached);
            }
            else if (modelType == "tree")
            {
                model = ml.Regression.Trainers.FastForest(
                        new FastForestRegressionTrainer.Options
                        {
                            LabelColumnName             = "Label",
                            FeatureColumnName           = "Features",
                            NumberOfTrees               = 1,
                            NumberOfLeaves              = 2048,
                            MinimumExampleCountPerLeaf  = 5,
                            FeatureFraction             = 1.0
                        })
                    .Fit(trainCached);
            }
            else
            {
                throw new ArgumentException($"Unknown modelType: {modelType}");
            }
        });

        double trainNs      = NowNs() - t1;
        long trainHeapDelta = GetHeapBytes() - trainHeapBefore;
        long trainRssDelta  = trainRssPeak;

        // --- Infer phase ---
        long inferHeapBefore = GetHeapBytes();
        double t2 = NowNs();

        IDataView predView = null!;
        long inferRssPeak = PeakRssBytesDuring(() =>
        {
            predView = model.Transform(testCached);
            ml.Data
                .CreateEnumerable<ModelOutput>(predView, reuseRowObject: false)
                .ToList();  // forces every prediction to actually execute
        });

        double inferNs      = NowNs() - t2;
        long inferHeapDelta = GetHeapBytes() - inferHeapBefore;
        long inferRssDelta  = inferRssPeak;
        
        // --- Accuracy (R²) — computed outside timed blocks so it does not affect timing ---
        var metrics = ml.Regression.Evaluate(predView, labelColumnName: "Label", scoreColumnName: "Score");
        double r2 = metrics.RSquared;
        double rmse = metrics.RootMeanSquaredError;


        return new PhaseResult(
            splitNs, splitHeapDelta, splitRssDelta,
            trainNs, trainHeapDelta, trainRssDelta,
            inferNs, inferHeapDelta, inferRssDelta, r2, rmse);
    }
    
    // -------------------------------------------------------------------------
// Polls WorkingSet64 on a background thread every ~5 ms and returns the
// peak observed. Catches native heap peaks (e.g. FastForest tree buffers)
// that a single before/after snapshot misses because the OS can page them
// out before the "after" snapshot is taken.
// -------------------------------------------------------------------------
    static long PeakRssBytesDuring(Action work)
    {
        long peak = 0;
        bool done = false;
        var proc = Process.GetCurrentProcess();

        var poller = new Thread(() =>
        {
            while (!done)
            {
                proc.Refresh();
                long ws = proc.WorkingSet64;
                if (ws > peak) peak = ws;
                Thread.Sleep(5);
            }
        }) { IsBackground = true };

        poller.Start();
        work();
        done = true;
        poller.Join();

        return peak;
    }

    class ModelOutput
    {
        [ColumnName("Score")]
        public float Score { get; set; }
    }
    class ModelInput
    {
        [VectorType]
        public float[] Features { get; set; } = default!;
        public float Label { get; set; }
    }

    class ResultRow
    {
        public string StratifyByTarget = "False";
        public string DeriveNetworkAsymmetry = "False";
        public string VarySpiltPerRepeat = "False";
        public string Language = "";
        public string Library = "";
        public string Model = "";
        public int SubsetSize;
        public int Repeat;
        public int NFeatures;
        public int Seed;
        public int SplitSeed;
        public double TestSize;
        // Load
        public double LoadNs;
        public long LoadHeapDeltaBytes;
        public long LoadRssDeltaBytes;
        // Clean
        public double CleanNs;
        public long CleanHeapDeltaBytes;
        public long CleanRssDeltaBytes;
        // Split
        public double SplitNs;
        public long SplitHeapDeltaBytes;
        public long SplitRssDeltaBytes;
        // Aggregate
        public double PreprocessNs;
        public double PipelineNs;

        // Train
        public double TrainNs;
        public long TrainHeapDeltaBytes;
        public long TrainRssDeltaBytes;
        // Infer
        public double InferNs;
        public long InferHeapDeltaBytes;
        public long InferRssDeltaBytes;
        // Total
        public double TotalNs;
        // Accuracy
        public double R2;
        public double Rmse;
    }
}