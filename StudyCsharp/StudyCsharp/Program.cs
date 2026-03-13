using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text.Json;

class Program
{
    const string TARGET = "compatibility_score";

    // Base features to load and consider (will drop constant ones later)
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


    static double NowNs() =>
        Stopwatch.GetTimestamp() * 1_000_000_000.0 / Stopwatch.Frequency;

    // Entry point which runs the experiment: load, clean, select features, 
    // then for each subset size: 
    // warmup, then repeat: split, train, infer, and record timings.
    static void Main(string[] args)
    {
        var solutionRoot = Directory.GetCurrentDirectory();
        string csvPath = @"C:\Code\Thesis\data\compatibility_pairs.csv";
        string outPath = "results/csharp_timings.csv";

        int seed = 42;
        float testSize = 0.2f;
        int[] sizes = { 5000, 10000, 25000, 50000 };
        int repeats = 100;
        int warmup = 1;

        if (!File.Exists(csvPath))
            throw new FileNotFoundException(csvPath);

        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);

        var (data, features, LoadNs, CleanNs) =
            LoadCleanAndSelect(csvPath);

        var results = new List<ResultRow>();

        foreach (var n in sizes)
        {
            int nEff = Math.Min(n, data.Count);
            var subset = data.Take(nEff).ToList();

            Console.WriteLine($"Total rows: {data.Count}");
            for (int w = 0; w < warmup; w++)
                SplitTrainInfer(subset, features, testSize, seed);

            for (int r = 0; r < repeats; r++)
            {
                var times = SplitTrainInfer(subset, features, testSize, seed);

                double preprocess = CleanNs + times.SplitNs;
                double total = LoadNs + CleanNs +
                               times.SplitNs + times.TrainNs + times.InferNs;

                results.Add(new ResultRow
                {
                    SubsetSize = nEff,
                    Repeat = r,
                    TrainNs = times.TrainNs,
                    InferNs = times.InferNs,
                    SplitNs = times.SplitNs,
                    TotalNs = total,
                    PreprocessNs = preprocess,
                    NFeatures = features.Count
                });
            }
        }

        using var writer = new StreamWriter(outPath);
        writer.WriteLine("subset_size,repeat,n_features,split_ns,train_ns,infer_ns,preprocess_ns,total_ns");

        foreach (var r in results)
        {
            writer.WriteLine($"{r.SubsetSize},{r.Repeat},{r.NFeatures}," +
                             $"{r.SplitNs},{r.TrainNs},{r.InferNs}," +
                             $"{r.PreprocessNs},{r.TotalNs}");
        }

        var meta = new
        {
            csv = Path.GetFullPath(csvPath),
            sizes_requested = sizes,
            repeats,
            warmup,
            seed,
            test_size = testSize,
            target = TARGET,
            base_features_requested = BASE_FEATURES,
            features_used = features,
            rows_after_cleaning = data.Count
        };

        File.WriteAllText(
            Path.ChangeExtension(outPath, ".meta.json"),
            JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true })
        );

        Console.WriteLine("Done.");
    }

    static (List<Dictionary<string, float>>, List<string>, double, double)
        LoadCleanAndSelect(string path)
    {
        double t0 = NowNs();

        List<Dictionary<string, float>> rows;

        using (var reader = new StreamReader(path))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            var records = csv.GetRecords<dynamic>().ToList();

            rows = records.Select(r =>
            {
                var rowDict = (IDictionary<string, object>)r;

                var dict = new Dictionary<string, float>();

                foreach (var f in BASE_FEATURES.Append(TARGET))
                {
                    if (rowDict.TryGetValue(f, out var raw))
                    {
                        var str = raw?.ToString()?.Trim();

                        if (float.TryParse(
                                str,
                                NumberStyles.Any,
                                CultureInfo.InvariantCulture,
                                out float val))
                        {
                            dict[f] = val;
                        }
                    }
                }

                return dict;
            }).ToList();
        }

        double LoadNs = NowNs() - t0;

        double t1 = NowNs();

        rows = rows
            .Where(r => BASE_FEATURES.Append(TARGET)
            .All(c => r.ContainsKey(c)))
            .ToList();

        var features = BASE_FEATURES.ToList();

        // Drop constant features
        features = features
            .Where(f => rows.Select(r => r[f]).Distinct().Count() > 1)
            .ToList();

        double CleanNs = NowNs() - t1;

        return (rows, features, LoadNs, CleanNs);
    }

    static (double SplitNs, double TrainNs, double InferNs)
        SplitTrainInfer(
        List<Dictionary<string, float>> data,
        List<string> features,
        float testSize,
        int seed)
    {
        double t0 = NowNs();

        var rnd = new Random(seed);
        var shuffled = data.OrderBy(_ => rnd.Next()).ToList();

        int splitIndex = (int)(shuffled.Count * (1 - testSize));

        var train = shuffled.Take(splitIndex).ToList();
        var test = shuffled.Skip(splitIndex).ToList();

        double SplitNs = NowNs() - t0;

        var ml = new MLContext(seed);

        var featureCount = features.Count;

        var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
        schemaDef[nameof(ModelInput.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, featureCount);

        var trainData = train.Select(r => new ModelInput
        {
            Features = features.Select(f => r[f]).ToArray(),
            Label = r[TARGET]
        }).ToList();

        var testData = test.Select(r => new ModelInput
        {
            Features = features.Select(f => r[f]).ToArray(),
            Label = r[TARGET]
        }).ToList();

        var trainView = ml.Data.LoadFromEnumerable(trainData, schemaDef);
        var testView = ml.Data.LoadFromEnumerable(testData, schemaDef);

        var pipeline = ml.Regression.Trainers.Sdca(
            labelColumnName: "Label",
            featureColumnName: "Features");

        double t1 = NowNs();

        var model = pipeline.Fit(trainView);

        double TrainNs = NowNs() - t1;

        double t2 = NowNs();

        var predictions = model.Transform(testView);

        double InferNs = NowNs() - t2;

        return (SplitNs, TrainNs, InferNs);
    }

    class ModelInput
    {
        [VectorType]   // <-- set correct feature count
        public float[] Features { get; set; } = default!;
        public float Label { get; set; }
    }

    class ResultRow
    {
        public int SubsetSize;
        public int Repeat;
        public int NFeatures;
        public double SplitNs;
        public double TrainNs;
        public double InferNs;
        public double PreprocessNs;
        public double TotalNs;
    }
}