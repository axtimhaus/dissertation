using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using parameter_study;
using Plotly.NET;
using RefraSin.Compaction.ProcessModel;
using RefraSin.Coordinates;
using RefraSin.Coordinates.Absolute;
using RefraSin.MaterialData;
using RefraSin.ParquetStorage;
using RefraSin.ParticleModel.ParticleFactories;
using RefraSin.ParticleModel.Remeshing;
using RefraSin.Plotting;
using RefraSin.ProcessModel;
using RefraSin.ProcessModel.Sintering;
using RefraSin.Storage;
using RefraSin.TEPSolver;
using Serilog;

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

var particle1 = new ShapeFunctionParticleFactory(
    input.Particle1.Radius,
    input.Particle1.PeakHeight,
    input.Particle1.PeakCount,
    input.Particle1.Ovality,
    material1Id
)
{
    CenterCoordinates = (input.Particle1.X, input.Particle1.Y),
    NodeCount = input.Particle1.NodeCount,
}.GetParticle(input.Particle1.Id);

var particle2 = new ShapeFunctionParticleFactory(
    input.Particle2.Radius,
    input.Particle2.PeakHeight,
    input.Particle2.PeakCount,
    input.Particle2.Ovality,
    material2Id
)
{
    CenterCoordinates = (input.Particle2.X, input.Particle2.Y),
    NodeCount = input.Particle2.NodeCount,
    RotationAngle = Angle.Half,
}.GetParticle(input.Particle2.Id);

var initialState = new SystemState(Guid.Empty, 0, [particle1, particle2]);

ParticlePlot.PlotParticles(initialState.Particles).SaveHtml("initialState.html");

var compactedState = new FocalCompactionStep(
    new AbsolutePoint(0, 0),
    input.Particle1.Radius / 100
).Solve(initialState);

ParticlePlot.PlotParticles(compactedState.Particles).SaveHtml("compactedState.html");

Log.Logger = new LoggerConfiguration().WriteTo.File("run.log").CreateLogger();
var loggerFactory = LoggerFactory.Create(builder => { builder.AddSerilog(); });

var solver = new SinteringSolver(loggerFactory, SolverRoutines.Default);

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

ParticlePlot.PlotParticles(finalState.Particles).SaveHtml("finalState.html");

storage.Dispose();
Log.CloseAndFlush();