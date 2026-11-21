/* Objective, gradient, and Adam update for the interactive demo. */
(function (root) {
    'use strict';

    function loss(x, y) {
        return 0.5 * (x * x + y * y) + 0.3 * Math.sin(3 * x) * Math.cos(3 * y) + 0.2 * (x + y);
    }

    function gradient(x, y) {
        return [
            x + 0.9 * Math.cos(3 * x) * Math.cos(3 * y) + 0.2,
            y - 0.9 * Math.sin(3 * x) * Math.sin(3 * y) + 0.2,
        ];
    }

    function step(state, {alpha, beta1, beta2, eps = 1e-8}) {
        const origin = state.theta.slice();
        const grad = gradient(...origin);
        const t = ++state.t;
        const firstCorrection = 1 - beta1 ** t;
        const secondCorrection = 1 - beta2 ** t;
        const mHat = [];
        const vHat = [];
        const delta = [];
        for (let axis = 0; axis < grad.length; axis++) {
            const g = grad[axis];
            state.m[axis] = beta1 * state.m[axis] + (1 - beta1) * g;
            state.v[axis] = beta2 * state.v[axis] + (1 - beta2) * g * g;
            mHat[axis] = state.m[axis] / firstCorrection;
            vHat[axis] = state.v[axis] / secondCorrection;
            delta[axis] = -alpha * mHat[axis] / (Math.sqrt(vHat[axis]) + eps);
            state.theta[axis] += delta[axis];
        }
        return {origin, grad, mHat, vHat, delta};
    }

    const adamCore = {loss, gradient, step};
    if (typeof module === 'object' && module.exports) module.exports = adamCore;
    else root.adamCore = adamCore;
})(globalThis);
