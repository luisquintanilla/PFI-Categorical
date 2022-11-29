// Add using statements
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Spectre.Console;

// Initialize MLContext
var ctx = new MLContext();

// Load data
var dataPath = @"C:\Datasets\RestaurantScores.csv";

Console.WriteLine("Loading dataset...");

var inferColumnInfo = ctx.Auto().InferColumns(dataPath,labelColumnName:"risk_category",separatorChar:',');

var loader = ctx.Data.CreateTextLoader(inferColumnInfo.TextLoaderOptions);
var dv = loader.Load(dataPath);

// Split into train / validation sets
var trainValSplit = ctx.Data.TrainTestSplit(dv, testFraction:0.2);

// Create pipeline
var pipeline = 
    ctx.Transforms.Conversion.MapValueToKey("Label", inferColumnInfo.ColumnInformation.LabelColumnName)
        .Append(ctx.Auto().Featurizer(dv, inferColumnInfo.ColumnInformation))
        .Append(ctx.Auto().MultiClassification(useLgbm:false))
        .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

// Initialize Experiment
var experiment = ctx.Auto().CreateExperiment();

// Configure experiment
experiment
    .SetDataset(trainValSplit)
    .SetPipeline(pipeline)
    .SetTrainingTimeInSeconds(90)
    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy,labelColumn: "Label");

// Run experiment
Console.WriteLine("Running experiment...");
var experimentResult = await experiment.RunAsync();

// Get best model
var bestModel = experimentResult.Model;

// Calculate PFI
var transformedData = bestModel.Transform(trainValSplit.TestSet);

Console.WriteLine("Calculating PFI...");

var pfi = 
    ctx.MulticlassClassification.PermutationFeatureImportance(bestModel, transformedData, permutationCount: 2);

Console.WriteLine("Calculating PFI Complete...");

// Take top 10 "features"
var pfiResults =
    pfi.Select(x => Tuple.Create(x.Key, x.Value.MacroAccuracy.Mean))
        .OrderByDescending(x => x.Item2);

DisplayTopNImportantFeatures(pfiResults);

Console.WriteLine("Program Complete...");

// Helper function to display PFI Results
void DisplayTopNImportantFeatures(IOrderedEnumerable<Tuple<string, double>> features, int n = 10) 
{
    //Create table
    var table = new Table();

    // Add columns
    table.AddColumn("Feature");
    table.AddColumn("Importance");

    //Add rows
    foreach (var feature in features)
    {
        table.AddRow(feature.Item1, feature.Item2.ToString());
    }

    AnsiConsole.Write(table);
};