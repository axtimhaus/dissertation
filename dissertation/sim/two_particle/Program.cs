using System.Text;
using System.Text.Json;
using parameter_study;
using Plotly.NET;
using RefraSin.Compaction.ProcessModel;
using RefraSin.Coordinates.Absolute;
using RefraSin.MaterialData;
using RefraSin.ParquetStorage;
using RefraSin.ParticleModel.Nodes;
using RefraSin.ParticleModel.ParticleFactories;
using RefraSin.ParticleModel.Particles;
using RefraSin.ParticleModel.Remeshing;
using RefraSin.Plotting;
using RefraSin.ProcessModel;
using RefraSin.ProcessModel.Sintering;
using RefraSin.Storage;
using RefraSin.TEPSolver;
using RefraSin.TEPSolver.StepWidthControllers;
using Serilog;

Log.Logger = new LoggerConfiguration().MinimumLevel.Information().WriteTo.File("run.log").WriteTo.Console().CreateLogger();

var inputFile = args[0];
var outputFile = args[1];

var inputText = File.ReadAllText(inputFile, encoding: Encoding.UTF8);

var input =
    JsonSerializer.Deserialize<Input>(
        inputText,
        new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower }
    ) ?? throw new ArgumentNullException("input");

var grainBoundary = new InterfaceProperties(
    input.GrainBoundary.DiffusionCoefficient,
    input.GrainBoundary.Energy
);

var material1Id = Guid.NewGuid();
var material2Id = Guid.NewGuid();

var material1 = new Material(
    material1Id,
    "material1",
    new BulkProperties(0, input.VacancyConcentration),
    new SubstanceProperties(input.Material1.Density, input.Material1.MolarMass),
    new InterfaceProperties(
        input.Material1.Surface.DiffusionCoefficient,
        input.Material1.Surface.Energy
    ),
    new Dictionary<Guid, IInterfaceProperties> { { material2Id, grainBoundary } }
);

var material2 = new Material(
    material2Id,
    "material2",
    new BulkProperties(0, input.VacancyConcentration),
    new SubstanceProperties(input.Material2.Density, input.Material1.MolarMass),
    new InterfaceProperties(
        input.Material2.Surface.DiffusionCoefficient,
        input.Material2.Surface.Energy
    ),
    new Dictionary<Guid, IInterfaceProperties> { { material1Id, grainBoundary } }
);

var particle1 = new ShapeFunctionParticleFactoryEllipseOvalityCosPeaks(
    material1Id,
    (input.Particle1.X, input.Particle1.Y),
    input.Particle1.RotationAngle,
    input.Particle1.NodeCount,
    input.Particle1.Radius,
    input.Particle1.Ovality,
    input.Particle1.PeakCount,
    input.Particle1.PeakHeight
).GetParticle(input.Particle1.Id);

var particle2 = new ShapeFunctionParticleFactoryEllipseOvalityCosPeaks(
    material2Id,
    (input.Particle2.X, input.Particle2.Y),
    input.Particle2.RotationAngle,
    input.Particle2.NodeCount,
    input.Particle2.Radius,
    input.Particle2.Ovality,
    input.Particle2.PeakCount,
    input.Particle2.PeakHeight
).GetParticle(input.Particle2.Id);

var initialState = new SystemState(Guid.Empty, 0, [particle1, particle2]);

ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(initialState.Particles).SaveHtml("initialState.html");

var compactedState = new FocalCompactionStep(
    new AbsolutePoint(0, 0),
    stepDistance: input.Particle1.Radius / 100,
    minimumRelativeIntrusion: 0.5,
    maxStepCount: 1
).Solve(initialState);

ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(compactedState.Particles).SaveHtml("compactedState.html");

var routines = SolverRoutines.Default with
{
    Remeshers = YieldRemeshers(),
    StepWidthController = new MaximumDisplacementAngleStepWidthController(maximumDisplacementAngle: input.TimeStepAngleLimit)
};

var solver = new SinteringSolver(routines, remeshingEverySteps: 50);

var plotHandler = new PlotEventHandler();
solver.SessionInitialized += plotHandler.HandleSessionInitialized;

var process = new SinteringStep(
    input.Duration,
    input.Temperature,
    solver,
    [material1, material2],
    input.GasConstant
);

var storage = new ParquetStorage(outputFile);
process.UseStorage(storage);

var finalState = process.Solve(compactedState);

ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(finalState.Particles).SaveHtml("finalState.html");

storage.Dispose();
Log.CloseAndFlush();

IEnumerable<IParticleSystemRemesher> YieldRemeshers()
{
    if (input.FreeSurfaceRemesherOptions is not null)
    {
        yield return new FreeSurfaceRemesher(
            input.FreeSurfaceRemesherOptions.DeletionLimit,
            input.FreeSurfaceRemesherOptions.AdditionLimit,
            input.FreeSurfaceRemesherOptions.MinWidthFactor,
            input.FreeSurfaceRemesherOptions.MaxWidthFactor,
            input.FreeSurfaceRemesherOptions.TwinPointLimit,
            input.FreeSurfaceRemesherOptions.NeckProtectionCount
        );
    }

    yield return new NeckNeighborhoodRemesher(input.NeckDeletionLimit);
}

class PlotEventHandler
{
    private int _counter;

    public void HandleSessionInitialized(object? sender, SinteringSolver.SessionInitializedEventArgs e)
    {
        ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(e.SolverSession.CurrentState.Particles)
            .SaveHtml($"session_{_counter}.html");
        _counter++;
    }

    public void HandleStepCalculated(object? sender, SinteringSolver.StepSuccessfullyCalculatedEventArgs e)
    {
        Chart.Combine([
                ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(e.OldState.Particles),
                ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(e.NewState.Particles)
            ])
            .SaveHtml($"step_{_counter}.html");
        _counter++;
    }
}
