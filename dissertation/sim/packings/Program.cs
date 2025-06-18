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

var materialId = Guid.NewGuid();

var material = new Material(
    materialId,
    "material",
    new BulkProperties(0, input.VacancyConcentration),
    new SubstanceProperties(input.Material.Density, input.Material.MolarMass),
    new InterfaceProperties(
        input.Material.Surface.DiffusionCoefficient,
        input.Material.Surface.Energy
    ),
    new Dictionary<Guid, IInterfaceProperties> { { materialId, grainBoundary } }
);

var particles = input.Particles.Select(p => new
    ShapeFunctionParticleFactoryEllipseOvalityCosPeaks(
    materialId,
    (p.X, p.Y),
    p.RotationAngle,
    p.NodeCount,
    p.Radius,
    p.Ovality,
    p.PeakCount,
    p.PeakHeight
).GetParticle(p.Id)).ToArray();

var initialState = new SystemState(Guid.Empty, 0, particles);

ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(initialState.Particles).SaveHtml("initialState.html");

var compactedState = new FocalCompactionStep(
    new AbsolutePoint(0, 0),
    stepDistance: 0,
    minimumRelativeIntrusion: 0.05,
    maxStepCount: 2
).Solve(initialState);

ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(compactedState.Particles).SaveHtml("compactedState.html");

var solver = new SinteringSolver(SolverRoutines.Default, remeshingEverySteps: 50);

var plotHandler = new PlotEventHandler();
solver.SessionInitialized += plotHandler.HandleSessionInitialized;

var process = new SinteringStep(
    input.Duration,
    input.Temperature,
    solver,
    [material],
    input.GasConstant
);

var storage = new ParquetStorage(outputFile);
process.UseStorage(storage);

var finalState = process.Solve(compactedState);

ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(finalState.Particles).SaveHtml("finalState.html");

storage.Dispose();
Log.CloseAndFlush();

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
