using Microsoft.ML.Data;

public class Input
{
    [LoadColumn(0)]
    public int Label {get;set;}

    [LoadColumn(1)]
    public float saturasyon {get;set;}
    
    [LoadColumn(2)]
    public float EC {get;set;}

    [LoadColumn(3)]
    public float pH {get;set;}

    [LoadColumn(4)]
    public float kirec {get;set;}

    [LoadColumn(5)]
    public float fosfor {get;set;}

    [LoadColumn(6)]
    public float potasyum {get;set;}

    [LoadColumn(7)]
    public float organikMadde {get;set;}
}

public class Output
{
    public int PredictedLabel {get;set;}
    public float[] Score {get;set;}
}