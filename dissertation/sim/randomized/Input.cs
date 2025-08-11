namespace parameter_study;

class ParticleInput
{
    public required Guid Id { get; set; }
    public required double X { get; set; }
    public required double Y { get; set; }
    public required double RotationAngle { get; set; }
    public required double Radius { get; set; }
    public required double Ovality { get; set; }
    public required int PeakCount { get; set; }
    public required double PeakHeight { get; set; }
    public required double PeakShift { get; set; }
    public required int NodeCount { get; set; }
    public required MaterialInput Material { get; set; }
    public required Dictionary<Guid, InterfaceInput> GrainBoundaries { get; set; }
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
    public required ParticleInput[] Particles { get; set; }

    public required double Temperature { get; set; }
    public required double GasConstant { get; set; }
    public required double Duration { get; set; }
}
