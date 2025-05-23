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

    public required double Temperature { get; set; }

    public required double VacancyConcentration { get; set; }

    public required double GasConstant { get; set; }

    public required double Duration { get; set; }

    public required FreeSurfaceRemesherOptions? FreeSurfaceRemesherOptions { get; set; }

    public required double NeckDeletionLimit { get; set; }

    public required double TimeStepAngleLimit { get; set; }
}

class FreeSurfaceRemesherOptions
{
    public required double DeletionLimit { get; set; }

    public required double AdditionLimit { get; set; }

    public required double MinWidthFactor { get; set; }

    public required double MaxWidthFactor { get; set; }

    public required double TwinPointLimit { get; set; }
}
