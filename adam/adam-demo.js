const { loss, step } = globalThis.adamCore;
const byId = (id) => document.getElementById(id);
const totalSteps = 60;
const start = [2.2, -1.8];

const playButton = byId("play");
const stepButton = byId("step");
const interval = byId("interval");
const controls = ["alpha", "beta1", "beta2"].map(byId);
const readout = Object.fromEntries(
    ["iteration", "loss", "theta", "gradient", "first-moment", "second-moment", "delta"]
        .map((id) => [id, byId(id)]),
);

let timer;
let params;
let state;
let lastStep;
let view;
let trajectory;

function number(value) {
    if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2);
    return value.toFixed(4);
}

function pair(values) {
    return values ? `(${values.map(number).join(", ")})` : "—";
}

function showStep() {
    readout.iteration.textContent = `${state.t} / ${totalSteps}`;
    readout.loss.textContent = number(loss(...state.theta));
    readout.theta.textContent = pair(state.theta);
    readout.gradient.textContent = pair(lastStep?.grad);
    readout["first-moment"].textContent = pair(lastStep?.mHat);
    readout["second-moment"].textContent = pair(lastStep?.vHat);
    readout.delta.textContent = pair(lastStep?.delta);
    view?.update(trajectory, lastStep);
}

function play(running) {
    clearInterval(timer);
    timer = running ? setInterval(advance, Number(interval.value)) : undefined;
    stepButton.disabled = state.t === totalSteps;
    playButton.textContent = running ? "Pause" : stepButton.disabled ? "Replay" : "Play";
}

function advance() {
    lastStep = step(state, params);
    trajectory.push(state.theta.slice());
    if (state.t === totalSteps) play(false);
    showStep();
}

function reset() {
    params = Object.fromEntries(controls.map((input) => [input.id, Number(input.value)]));
    state = { theta: start.slice(), m: [0, 0], v: [0, 0], t: 0 };
    lastStep = null;
    trajectory = [start];
    play(false);
    showStep();
}

controls.forEach((input, index) => {
    input.addEventListener("input", () => {
        byId(`${input.id}-value`).textContent = Number(input.value).toFixed(index === 2 ? 4 : 3);
        reset();
    });
});

interval.addEventListener("input", () => {
    byId("interval-value").textContent = `${interval.value} ms`;
    if (timer !== undefined) play(true);
});

playButton.addEventListener("click", () => {
    if (state.t === totalSteps) reset();
    play(timer === undefined);
});

stepButton.addEventListener("click", () => {
    play(false);
    advance();
});

byId("reset").addEventListener("click", reset);
document.addEventListener("visibilitychange", () => {
    if (document.hidden && timer !== undefined) play(false);
});

byId("parameters").open = matchMedia("(min-width: 52rem) and (pointer: fine)").matches;
reset();

const plotMessage = byId("plot-message");
import("./adam-view.js").then(({ createView }) => {
    view = createView(byId("plot"), loss, totalSteps);
    plotMessage.hidden = true;
    showStep();
}).catch((error) => {
    console.warn("Could not initialize the Adam plot.", error);
    byId("plot-guide").hidden = true;
    plotMessage.textContent = "The 3D plot is unavailable. You can still run Adam and inspect the step values.";
});
