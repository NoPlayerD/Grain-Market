using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data.DataView;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers.FastTree;
/* VERSIONS
Microsoft.ML.AutoML 0.20.1
Microsoft.ML 2.0.1
*/
#region AYAR

Console.Title = "Pazarcı";
#endregion

#region Veriler

Log("MLContext oluşturuluyor..");
MLContext context = new MLContext();

Log("Veri yükleniyor..");
IDataView data = context.Data.LoadFromTextFile<Input>(@"..\..\..\..\dataset.csv", hasHeader: true, separatorChar:',');
#endregion

#region Veri ayırma

Log("Eğitim ve test verisi ayrılıyor..");
var split = context.Data.TrainTestSplit(data, testFraction:0.2);
var trainData = split.TrainSet;
var testData = split.TestSet;
#endregion

#region Pipeline ve Model oluşturma

Log("Boruhattı oluşturuluyor..");
var pipeline = context.Transforms.Concatenate("Features", new[]{"saturasyon", "EC", "pH", "kirec", "fosfor", "potasyum", "organikMadde"})
    .Append(context.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName:"Label"))
    .Append(context.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator:context.BinaryClassification.Trainers.FastTree()))   
    .Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName:"PredictedLabel", inputColumnName:"PredictedLabel"));

Log("Model oluşturuluyor..");
var model = pipeline.Fit(data);
#endregion

#region Değerlendirme

Log("Değerlendirme verisi oluşturuluyor..:");
Reset();
var sampleData = new Input()
{
    saturasyon = 39,
    EC = 4.5f,
    pH = 7.4f,
    kirec = -15,
    fosfor = 35,
    potasyum = 203,
    organikMadde = 3
};
Console.WriteLine($"Satursayon: {sampleData.saturasyon}\nEC: {sampleData.EC}\npH: {sampleData.pH}\nKirec: {sampleData.kirec}\nFosfor: {sampleData.fosfor}\nPotasyum: {sampleData.potasyum}\nOrganik Madde: {sampleData.organikMadde}", Console.ForegroundColor = ConsoleColor.DarkBlue);

Log("Dönüştürme yapılıyor..");
var predictions = model.Transform(testData);
Log("Değerlendiriliyor..:");
var evaluate = context.MulticlassClassification.Evaluate(predictions);

Reset();
Console.WriteLine($"Macro Accuary: {evaluate.MacroAccuracy}\nMicro Accuary: {evaluate.MicroAccuracy}", Console.ForegroundColor = ConsoleColor.DarkBlue);
#endregion

#region Tahmin etme

Log("Tahmin motoru oluşturuluyor..");
var predictionFunc = context.Model.CreatePredictionEngine<Input, Output>(model);
Log("Tahmin işlemi gerçekleştiriliyor..:");
var result = predictionFunc.Predict(sampleData);
Reset();
Console.WriteLine("================================================", Console.BackgroundColor = ConsoleColor.White, Console.ForegroundColor = ConsoleColor.Black );
Reset();
Console.WriteLine($"Tahmin edilen Değer: {result.PredictedLabel}\nSkor: {result.Score[0]}");
#endregion

#region DİĞER
void Log(string msg)
{
    Console.WriteLine($"LOG: {msg}", Console.ForegroundColor = ConsoleColor.Cyan);
}
void Reset()
{
   Console.ResetColor();
}
Console.WriteLine("==================== FINITO ====================", Console.BackgroundColor = ConsoleColor.White,Console.ForegroundColor = ConsoleColor.Black);
Console.ReadKey();
#endregion