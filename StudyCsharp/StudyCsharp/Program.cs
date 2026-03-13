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


    static double NowMs() =>
        Stopwatch.GetTimestamp() * 1000.0 / Stopwatch.Frequency;

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

        var (data, features, loadMs, cleanMs) =
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

                double preprocess = cleanMs + times.SplitMs;
                double total = loadMs + cleanMs +
                               times.SplitMs + times.TrainMs + times.InferMs;

                results.Add(new ResultRow
                {
                    SubsetSize = nEff,
                    Repeat = r,
                    TrainMs = times.TrainMs,
                    InferMs = times.InferMs,
                    SplitMs = times.SplitMs,
                    TotalMs = total,
                    PreprocessMs = preprocess,
                    NFeatures = features.Count
                });
            }
        }

        using var writer = new StreamWriter(outPath);
        writer.WriteLine("subset_size,repeat,n_features,split_ms,train_ms,infer_ms,preprocess_ms,total_ms");

        foreach (var r in results)
        {
            writer.WriteLine($"{r.SubsetSize},{r.Repeat},{r.NFeatures}," +
                             $"{r.SplitMs},{r.TrainMs},{r.InferMs}," +
                             $"{r.PreprocessMs},{r.TotalMs}");
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
        double t0 = NowMs();

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

        double loadMs = NowMs() - t0;

        double t1 = NowMs();

        rows = rows
            .Where(r => BASE_FEATURES.Append(TARGET)
            .All(c => r.ContainsKey(c)))
            .ToList();

        var features = BASE_FEATURES.ToList();

        // Drop constant features
        features = features
            .Where(f => rows.Select(r => r[f]).Distinct().Count() > 1)
            .ToList();

        double cleanMs = NowMs() - t1;

        return (rows, features, loadMs, cleanMs);
    }

    static (double SplitMs, double TrainMs, double InferMs)
        SplitTrainInfer(
        List<Dictionary<string, float>> data,
        List<string> features,
        float testSize,
        int seed)
    {
        double t0 = NowMs();

        var rnd = new Random(seed);
        var shuffled = data.OrderBy(_ => rnd.Next()).ToList();

        int splitIndex = (int)(shuffled.Count * (1 - testSize));

        var train = shuffled.Take(splitIndex).ToList();
        var test = shuffled.Skip(splitIndex).ToList();

        double splitMs = NowMs() - t0;

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

        double t1 = NowMs();

        var model = pipeline.Fit(trainView);

        double trainMs = NowMs() - t1;

        double t2 = NowMs();

        var predictions = model.Transform(testView);

        double inferMs = NowMs() - t2;

        return (splitMs, trainMs, inferMs);
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
        public double SplitMs;
        public double TrainMs;
        public double InferMs;
        public double PreprocessMs;
        public double TotalMs;
    }
}