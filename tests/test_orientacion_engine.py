import unittest

from orientacion_engine import (
    INTERESTS,
    PATHS,
    evaluate_expert_rules,
    predict_with_model,
    train_demo_models,
    validate_profile,
)


def profile_with(**overrides):
    profile = {key: 0 for key in INTERESTS}
    profile.update(overrides)
    return profile


class OrientationEngineTests(unittest.TestCase):
    def test_expert_rules_explain_a_matching_path(self):
        results = evaluate_expert_rules(
            profile_with(exterior=5, ciencia=4, tecnologia=3)
        )

        self.assertEqual(results[0].path, "Medio ambiente")
        self.assertTrue(results[0].evidence)

    def test_no_rules_fire_for_zero_interest_profile(self):
        self.assertEqual(evaluate_expert_rules(profile_with()), [])

    def test_profile_rejects_missing_keys_and_out_of_range_values(self):
        with self.assertRaises(ValueError):
            validate_profile({"tecnologia": 3})
        with self.assertRaises(ValueError):
            validate_profile(profile_with(tecnologia=6))

    def test_tree_and_forest_return_all_paths_with_sorted_votes(self):
        decision_tree, forest = train_demo_models(samples_per_path=8)

        for model in (decision_tree, forest):
            results = predict_with_model(model, profile_with(tecnologia=5, ciencia=4))
            self.assertEqual({path for path, _ in results}, set(PATHS))
            self.assertEqual(results, sorted(results, key=lambda item: (-item[1], item[0])))
            self.assertAlmostEqual(sum(probability for _, probability in results), 1.0)


if __name__ == "__main__":
    unittest.main()
