"""CPU regression tests with small synthetic modules, not trained-model accuracy."""
import copy
import unittest
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from wham_coreml import WHAMInit, WHAMStep, check_wiring, feedback, load_weights, reference_step
from extractViT import RGBImageStudent

MAIN_JOINTS = list(range(10)) + list(range(12, 22))


class NeuralInit(nn.Module):
    def __init__(self, width, hidden, layers=2):
        super().__init__()
        self.linear = nn.Linear(width, 2 * layers * hidden)
        self.layers = layers
        self.hidden = hidden

    def forward(self, x):
        state = self.linear(x).reshape(x.shape[0], 2, self.layers, self.hidden)
        state = state.permute(1, 2, 0, 3).contiguous()
        return state[0], state[1]


class Regressor(nn.Module):
    def __init__(self, width, hidden, outputs):
        super().__init__()
        self.rnn = nn.LSTM(width, hidden, num_layers=2, batch_first=True)
        self.heads = nn.ModuleList(nn.Linear(hidden, size) for size in outputs)

    def forward(self, x, previous, state):
        hidden, state = self.rnn(torch.cat([x, *previous], -1), state)
        return [head(hidden) for head in self.heads], hidden, state


class Integrator(nn.Module):
    """Same operations/masking as upstream, with much smaller hidden layers."""
    def __init__(self, context):
        super().__init__()
        self.layer1 = nn.Linear(context + 1024, 24)
        self.relu1 = nn.ReLU()
        self.dr1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(24, 24)
        self.relu2 = nn.ReLU()
        self.dr2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(24, context)

    def forward(self, x, features):
        mask = (features != 0).all(dim=-1).all(dim=-1)
        out = self.dr1(self.relu1(self.layer1(torch.cat((x, features), -1))))
        out = self.dr2(self.relu2(self.layer2(out)))
        out = self.layer3(out)
        out[mask] = out[mask] + x[mask]
        return out


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        hidden, context = 8, 59
        self.motion_encoder = nn.Module()
        self.motion_encoder.embed_layer = nn.Linear(37, hidden)
        self.motion_encoder.pos_drop = nn.Dropout(0.1)
        self.motion_encoder.neural_init = NeuralInit(88, hidden)
        self.motion_encoder.regressor = Regressor(hidden + 51, hidden, [51])
        self.trajectory_decoder = nn.Module()
        self.trajectory_decoder.regressor = Regressor(context + 12, context, [3, 6])
        self.integrator = Integrator(context)
        self.motion_decoder = nn.Module()
        self.motion_decoder.neural_init = NeuralInit(120, context)
        self.motion_decoder.regressor = Regressor(context + 144, context, [144, 10, 3, 4])


class ImageFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)
        self.network = Network().eval()
        self.init = WHAMInit(self.network, MAIN_JOINTS).eval()
        self.step = WHAMStep(self.network).eval()
        self.initial = (torch.randn(1, 1, 88), torch.randn(1, 1, 144))
        with torch.no_grad():
            self.args = (torch.randn(1, 1, 37), torch.ones(1, 1, 1024),
                         torch.zeros(1, 1, 6), self.initial[0][..., :51],
                         torch.zeros(1, 1, 6), self.initial[1], *self.init(*self.initial))

    def assert_outputs_equal(self, actual, expected):
        self.assertEqual(len(actual), 13)
        for got, wanted in zip(actual, expected):
            torch.testing.assert_close(got, wanted, rtol=1e-5, atol=1e-6)

    def test_features_change_pose_shape_but_not_encoder_or_trajectory(self):
        deltas = check_wiring(self.network, self.step, self.args)
        for value in deltas.values():
            self.assertGreater(value, 0)

    def test_reference_nonzero_features(self):
        self.assert_outputs_equal(self.step(*self.args), reference_step(self.network, self.args))

    def test_reference_zero_features(self):
        args = (self.args[0], torch.zeros_like(self.args[1]), *self.args[2:])
        self.assert_outputs_equal(self.step(*args), reference_step(self.network, args))

    def test_reference_one_zero_feature_coordinate(self):
        features = self.args[1].clone()
        features[..., 10] = 0
        args = (self.args[0], features, *self.args[2:])
        self.assert_outputs_equal(self.step(*args), reference_step(self.network, args))

    def test_exact_upstream_residual_condition_over_batch_and_time(self):
        context = torch.randn(3, 4, 59)
        features = torch.ones(3, 4, 1024)
        features[1] = 0
        features[2, 2, 100] = 0
        torch.testing.assert_close(self.step.integrate(context, features),
                                   self.network.integrator(context, features))

    def test_init_uses_actual_lstm_layer_count(self):
        states = self.init(*self.initial)
        self.assertEqual(states[2].shape, (2, 1, 59))
        self.assertEqual(torch.count_nonzero(states[2]).item(), 0)
        self.assertEqual(torch.count_nonzero(states[3]).item(), 0)

    def test_integrator_is_the_loaded_module_not_random_replacement(self):
        self.assertIs(self.step.integrator, self.network.integrator)
        self.assertIs(self.step.integrator.layer1.weight, self.network.integrator.layer1.weight)

    def test_tracing_does_not_freeze_zero_feature_mask(self):
        with torch.no_grad():
            traced = torch.jit.trace(self.step, self.args)
            for features in (torch.ones_like(self.args[1]), torch.zeros_like(self.args[1])):
                args = (self.args[0], features, *self.args[2:])
                self.assert_outputs_equal(traced(*args), self.step(*args))

    def test_sequence_carries_each_backends_own_states(self):
        got_args, expected_args = self.args, self.args
        with torch.no_grad():
            for frame in range(12):
                x = torch.randn_like(self.args[0])
                features = torch.randn_like(self.args[1])
                if frame % 3 == 0:
                    features.zero_()
                got_args = (x, features, *got_args[2:])
                expected_args = (x, features, *expected_args[2:])
                got = self.step(*got_args)
                expected = reference_step(self.network, expected_args)
                self.assert_outputs_equal(got, expected)
                got_args = feedback(got_args, got, x, features)
                expected_args = feedback(expected_args, expected, x, features)

    def test_traced_sequence_with_independent_feedback(self):
        with torch.no_grad():
            traced = torch.jit.trace(self.step, self.args)
            traced_args, eager_args = self.args, self.args
            for frame in range(8):
                x, features = torch.randn_like(self.args[0]), torch.randn_like(self.args[1])
                if frame == 4:
                    features.zero_()
                traced_args, eager_args = (x, features, *traced_args[2:]), (x, features, *eager_args[2:])
                got, expected = traced(*traced_args), self.step(*eager_args)
                self.assert_outputs_equal(got, expected)
                traced_args = feedback(traced_args, got, x, features)
                eager_args = feedback(eager_args, expected, x, features)

    def test_missing_integrator_weight_is_rejected(self):
        weights = dict(self.network.state_dict())
        del weights["integrator.layer1.weight"]
        with self.assertRaises(RuntimeError):
            load_weights(self.network, {"model": weights})

    def test_only_smpl_weights_are_ignored(self):
        weights = dict(self.network.state_dict())
        weights["smpl.v_template"] = torch.zeros(1)
        load_weights(copy.deepcopy(self.network), {"model": weights})
        weights["unexpected.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            load_weights(self.network, {"model": weights})

    def test_wrong_feature_width_is_rejected(self):
        self.network.integrator.layer1 = nn.Linear(59 + 2048, 24)
        with self.assertRaises(ValueError):
            WHAMStep(self.network)

    def test_checkpoint_without_model_state_is_rejected(self):
        with self.assertRaises(ValueError):
            load_weights(self.network, {})


class RGBNormalizationTests(unittest.TestCase):
    def test_matches_training_normalization_for_rgb_pixels(self):
        pixels = torch.tensor([0, 128, 255], dtype=torch.float32).view(1, 3, 1, 1)
        expected = (pixels / 255 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        expected /= torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        torch.testing.assert_close(RGBImageStudent(nn.Identity())(pixels), expected)

    def test_normalization_survives_trace(self):
        model = RGBImageStudent(nn.Identity()).eval()
        example = torch.rand(1, 3, 8, 8) * 255
        traced = torch.jit.trace(model, example)
        for value in (0, 255):
            pixels = torch.full_like(example, value)
            torch.testing.assert_close(traced(pixels), model(pixels))


if __name__ == "__main__":
    unittest.main()
