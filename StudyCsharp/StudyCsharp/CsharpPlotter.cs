namespace StudyCsharp;

using System.Globalization;
using ScottPlot;

public class CsharpPlotter
{
    public static void Run()
    {
        var lines = File.ReadAllLines("results/csharp_timings.csv").Skip(1);

        var data = lines.Select(l =>
        {
            var p = l.Split(',');
            return new
            {
                subset = int.Parse(p[3]),
                model = p[2],
                total = double.Parse(p[15], CultureInfo.InvariantCulture) / 1e9
            };
        }).ToList();

        var grouped = data
            .GroupBy(x => new { x.subset, x.model })
            .Select(g => new
            {
                g.Key.subset,
                g.Key.model,
                avg = g.Average(x => x.total)
            })
            .ToList();

        var plt = new ScottPlot.Plot();

        foreach (var model in grouped.Select(g => g.model).Distinct())
        {
            var sub = grouped.Where(g => g.model == model).OrderBy(g => g.subset);

            double[] xs = sub.Select(x => (double)x.subset).ToArray();
            double[] ys = sub.Select(x => x.avg).ToArray();

            plt.Add.Scatter(xs, ys, label: model);
        }

        plt.Title("Total Time vs Dataset Size");
        plt.XLabel("Subset size");
        plt.YLabel("Time (seconds)");
        plt.Legend();

        plt.SavePng("results/total_time.png", 800, 600);
    }
}