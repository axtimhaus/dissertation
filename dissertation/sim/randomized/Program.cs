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
using RefraSin.ParticleModel.System;
using RefraSin.Plotting;
using RefraSin.ProcessModel;
using RefraSin.ProcessModel.Sintering;
using RefraSin.Storage;
using RefraSin.TEPSolver;
using RefraSin.TEPSolver.BreakConditions;
using Serilog;
using Serilog.Formatting.Display;

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Information()
    .WriteTo.File("run.log")
    .WriteTo.Console()
    .CreateLogger();

const string inputFile = "input.json";
const string outputFile = "output.parquet";

var inputText = File.ReadAllText(inputFile, encoding: Encoding.UTF8);

var input =
    JsonSerializer.Deserialize<Input>(
        inputText,
        new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower }
    ) ?? throw new ArgumentNullException("input");

var materials = input
    .Particles.Select(p => new ParticleMaterial(
        p.Id,
        "material",
        SubstanceProperties.FromDensityAndMolarMass(p.Material.Density, p.Material.MolarMass),
        new InterfaceProperties(p.Material.Surface.DiffusionCoefficient, p.Material.Surface.Energy),
        p.GrainBoundaries.Select(kvp => new KeyValuePair<Guid, IInterfaceProperties>(
                kvp.Key,
                new InterfaceProperties(kvp.Value.DiffusionCoefficient, kvp.Value.Energy / 2)
            ))
            .ToDictionary()
    ))
    .ToArray();

var particles = input
    .Particles.Select(p =>
        new ShapeFunctionParticleFactoryEllipseOvalityCosPeaks(
            p.Id,
            (p.X, p.Y),
            p.RotationAngle,
            p.NodeCount,
            p.Radius,
            p.Ovality,
            p.PeakCount,
            p.PeakHeight,
            p.PeakShift
        ).GetParticle(p.Id)
    )
    .ToArray();

var initialState = new SystemState(Guid.Empty, 0, particles);

ParticlePlot
    .PlotParticles<IParticle<IParticleNode>, IParticleNode>(initialState.Particles)
    .SaveHtml("initialState.html");

var compactionStep = new OneByOneCompactionStep(
    stepDistance: 2e-6,
    minimumIntrusion: 1.5e-6,
    maxStepCount: 10000
);

var plotHandler = new PlotEventHandler();

// compactionStep.SystemStateReported += plotHandler.HandleReportSystemState;

var compactedState = compactionStep.Solve(initialState);

ParticlePlot
    .PlotParticles<IParticle<IParticleNode>, IParticleNode>(compactedState.Particles)
    .SaveHtml("compactedState.html");

if (compactedState.Nodes.Where(n => n.Type is NodeType.GrainBoundary).Count() / 2 < 3)
    throw new Exception("contact creation failed, too few grain boundaries present");

var solver = new SinteringSolver(
    SolverRoutines.Default with
    {
        Remeshers =
        [
            new FreeSurfaceRemesher(deletionLimit: 0.15, neckProtectionCount: 10),
            new NeckNeighborhoodRemesher(),
            new LastSurfaceNodeRemesher(),
        ],
        BreakConditions = [new PoreClosedCondition(20e-6 / input.Particles[0].Radius)],
    },
    remeshingEverySteps: 50
);

var remeshedState = new SystemState(
    Guid.NewGuid(),
    compactedState.Time,
    solver.Routines.Remeshers.Aggregate<
        IParticleSystemRemesher,
        IParticleSystem<IParticle<IParticleNode>, IParticleNode>
    >(compactedState, (state, remesher) => remesher.RemeshSystem(state))
);

ParticlePlot
    .PlotParticles<IParticle<IParticleNode>, IParticleNode>(remeshedState.Particles)
    .SaveHtml("remeshedState.html");

plotHandler = new PlotEventHandler();
solver.SessionInitialized += plotHandler.HandleSessionInitialized;

var process = new SinteringStep(
    input.Duration,
    input.Temperature,
    solver,
    materials,
    input.GasConstant
);

var storage = new ParquetStorage(outputFile);
process.UseStorage(storage);

try
{
    var finalState = process.Solve(remeshedState);
    ParticlePlot
        .PlotParticles<IParticle<IParticleNode>, IParticleNode>(finalState.Particles)
        .SaveHtml("finalState.html");
}
finally
{
    storage.Dispose();
    Log.CloseAndFlush();
}

class PlotEventHandler
{
    private int _counter;

    public void HandleSessionInitialized(
        object? sender,
        SinteringSolver.SessionInitializedEventArgs e
    )
    {
        ParticlePlot
            .PlotParticles<IParticle<IParticleNode>, IParticleNode>(
                e.SolverSession.CurrentState.Particles
            )
            .SaveHtml($"session_{_counter}.html");
        _counter++;
    }

    public void HandleStepCalculated(
        object? sender,
        SinteringSolver.StepSuccessfullyCalculatedEventArgs e
    )
    {
        Chart
            .Combine(
                [
                    ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(
                        e.OldState.Particles
                    ),
                    ParticlePlot.PlotParticles<IParticle<IParticleNode>, IParticleNode>(
                        e.NewState.Particles
                    ),
                ]
            )
            .SaveHtml($"step_{_counter}.html");
        _counter++;
    }

    public void HandleReportSystemState(object? sender, IProcessStep.SystemStateReportedEventArgs e)
    {
        ParticlePlot
            .PlotParticles<IParticle<IParticleNode>, IParticleNode>(e.State.Particles)
            .SaveHtml($"state_{_counter}.html");
        _counter++;
    }
}
