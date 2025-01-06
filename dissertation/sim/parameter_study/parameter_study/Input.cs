namespace parameter_study;

class ParticleInput
{
    public required double X { get; set; }
    public required double Y { get; set; }
    public required double Radius { get; set; }
    public required double Ovality { get; set; }
    public required uint PeakCount { get; set; }
    public required double PeakHeight { get; set; }
    public required int NodeCount { get; set; }
}

class InterfaceInput
{
    public required double Energy { get; set; }
    public required double DiffusionCoefficient { get; set; }
}

class MaterialInput
{
    public required double Density { get; set; }
    public required double MolarMass { get; set; }
    public required InterfaceInput Surface { get; set; }
}

class Input
{
    public required ParticleInput Particle1 { get; set; }
    public required ParticleInput Particle2 { get; set; }
    public required MaterialInput Material1 { get; set; }
    public required MaterialInput Material2 { get; set; }
    public required InterfaceInput GrainBoundary { get; set; }
}
