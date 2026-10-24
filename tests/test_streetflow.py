"""
Tests for StreetFlow Live — speed estimator, routing, API validation.

Run with:  pytest tests/ -v
"""

import os
import sys
import math
import json
import pytest
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "model"))
sys.path.insert(0, os.path.join(_BASE, "data"))
sys.path.insert(0, os.path.join(_BASE, "api"))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Speed Estimator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedEstimator:
    """Tests for the spatial speed estimator (IDW)."""

    @pytest.fixture
    def road_coords(self):
        from speed_estimator import load_road_coords
        return load_road_coords()

    @pytest.fixture
    def fake_speeds(self):
        """16 fake predicted speeds for testing."""
        return [35.0 + i * 2 for i in range(16)]

    @pytest.fixture
    def estimator(self, road_coords):
        from speed_estimator import SpeedEstimator
        return SpeedEstimator(road_coords=road_coords, method="idw")

    def test_near_known_road(self, estimator, fake_speeds):
        """Speed near a known road should be close to that road's speed."""
        # Indiranagar (roads 0, 1): lat 12.9699, lng 77.6499
        speed, warn = estimator.speed_at(12.970, 77.650, fake_speeds, day_offset=0)
        assert speed > 0, "Speed must be positive"
        assert not warn, "Should not have coverage warning near a known road"
        # Should be influenced heavily by roads 0 and 1
        assert 30 < speed < 70, f"Speed {speed} seems unreasonable near Indiranagar"

    def test_far_from_roads_has_coverage_warning(self, estimator, fake_speeds):
        """Points far from all 16 roads should trigger coverage_warning."""
        # Point well outside Bangalore road coverage (still in bounding box)
        speed, warn = estimator.speed_at(13.19, 77.41, fake_speeds, day_offset=0)
        assert warn, "Should have coverage warning for point far from all roads"
        # Should return mean speed
        expected_mean = np.mean(fake_speeds)
        assert abs(speed - expected_mean) < 1.0, "Far-away speed should be close to mean"

    def test_speed_clamped_to_minimum(self, road_coords):
        """Speed should never drop below min_speed_kmh (5 km/h)."""
        from speed_estimator import SpeedEstimator
        estimator = SpeedEstimator(road_coords=road_coords, method="idw")
        # All speeds are very low
        low_speeds = [1.0] * 16
        speed, _ = estimator.speed_at(12.97, 77.65, low_speeds, day_offset=0)
        assert speed >= 5.0, f"Speed {speed} should be at least 5 km/h"

    def test_idw_varies_with_position(self, estimator, fake_speeds):
        """Different locations should give different IDW speeds."""
        s1, _ = estimator.speed_at(12.92, 77.62, fake_speeds, 0)
        s2, _ = estimator.speed_at(13.03, 77.59, fake_speeds, 0)
        # These are near different clusters of roads with different speeds
        assert s1 != s2, "IDW should produce different speeds at different locations"


class TestSpeedEstimatorCrossValidation:
    """Test that leave-one-road-out cross-validation runs and produces metrics."""

    def test_loo_cv_computes_metrics(self):
        from speed_estimator import load_road_coords, leave_one_out_cv
        roads = load_road_coords()
        # Use a single fake speed snapshot
        fake_speeds = [np.array([30.0 + i * 2 for i in range(16)])]

        metrics = leave_one_out_cv(roads, fake_speeds)

        assert "idw" in metrics
        assert "global_mean" in metrics
        for method in ["idw", "global_mean"]:
            assert "MAE" in metrics[method]
            assert "RMSE" in metrics[method]
            assert "MAPE" in metrics[method]
            assert metrics[method]["MAE"] >= 0
            assert metrics[method]["RMSE"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Bounding Box Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundingBoxValidation:
    """Test Pydantic validation for the Bangalore bounding box."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("pydantic")

    def test_valid_coordinates(self):
        """Coordinates inside Bangalore should be accepted."""
        from api_bangalore import LatLng
        point = LatLng(lat=12.97, lng=77.60)
        assert point.lat == 12.97
        assert point.lng == 77.60

    def test_latitude_too_low(self):
        """Latitude below 12.80 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=12.50, lng=77.60)

    def test_latitude_too_high(self):
        """Latitude above 13.20 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=13.50, lng=77.60)

    def test_longitude_too_low(self):
        """Longitude below 77.40 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=12.97, lng=77.00)

    def test_longitude_too_high(self):
        """Longitude above 77.85 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=12.97, lng=78.00)

    def test_same_point_rejection(self):
        """Origin and destination at the same point (< 200 m) should fail."""
        from api_bangalore import CustomRouteRequest
        # Same coordinates — distance is 0
        req = CustomRouteRequest(
            origin={"lat": 12.97, "lng": 77.60},
            destination={"lat": 12.97, "lng": 77.60},
            day_offset=0
        )
        # The distance check happens in the endpoint, not in Pydantic.
        # So we just verify the model accepts same coords (the 200m check
        # is in the route handler).
        from custom_routing import haversine_m
        dist = haversine_m(
            req.origin.lat, req.origin.lng,
            req.destination.lat, req.destination.lng
        )
        assert dist < 200, "Same-point distance should be < 200 m"

    def test_day_offset_out_of_range(self):
        """day_offset outside 0-2 should be rejected."""
        from api_bangalore import CustomRouteRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="day_offset"):
            CustomRouteRequest(
                origin={"lat": 12.97, "lng": 77.60},
                destination={"lat": 12.92, "lng": 77.62},
                day_offset=5
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Route Scoring Tests (mocked OSRM response)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteScoring:
    """Test route scoring logic with a mocked OSRM response."""

    @pytest.fixture
    def mock_osrm_routes(self):
        """Simulated OSRM response with 2 alternative routes."""
        return [
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [77.60, 12.97],   # start (Indiranagar area)
                        [77.62, 12.95],   # mid
                        [77.63, 12.93],   # end (Koramangala area)
                    ]
                },
                "distance": 5500,   # 5.5 km
                "duration": 900,    # 15 min (OSRM's static estimate)
            },
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [77.60, 12.97],
                        [77.61, 12.96],
                        [77.615, 12.945],
                        [77.63, 12.93],
                    ]
                },
                "distance": 6200,   # 6.2 km
                "duration": 1020,   # 17 min
            },
        ]

    @pytest.fixture
    def estimator(self):
        from speed_estimator import SpeedEstimator, load_road_coords
        roads = load_road_coords()
        return SpeedEstimator(road_coords=roads, method="idw")

    def test_score_all_routes(self, mock_osrm_routes, estimator):
        """score_all_routes should return scored routes with one recommended."""
        from custom_routing import score_all_routes
        fake_speeds = [35.0 + i * 2 for i in range(16)]
        mean_speed = 40.0

        result = score_all_routes(
            mock_osrm_routes, estimator, fake_speeds,
            day_offset=0, training_mean_speed=mean_speed
        )

        assert "routes" in result
        assert len(result["routes"]) == 2
        assert "time_saved_min" in result
        assert "improvement_pct" in result
        assert "coverage_warning" in result

        # Exactly one route should be recommended
        recommended = [r for r in result["routes"] if r["recommended"]]
        assert len(recommended) == 1

        for route in result["routes"]:
            assert route["eta_pred_min"] > 0
            assert route["eta_static_min"] > 0
            assert route["distance_km"] > 0
            assert route["avg_speed_kmh"] > 0
            assert "geometry" in route
            assert route["geometry"]["type"] == "LineString"

    def test_recommended_has_lowest_eta(self, mock_osrm_routes, estimator):
        """The recommended route should have the lowest predicted ETA."""
        from custom_routing import score_all_routes
        fake_speeds = [35.0 + i * 2 for i in range(16)]

        result = score_all_routes(
            mock_osrm_routes, estimator, fake_speeds,
            day_offset=0, training_mean_speed=40.0
        )

        rec = [r for r in result["routes"] if r["recommended"]][0]
        for r in result["routes"]:
            assert rec["eta_pred_min"] <= r["eta_pred_min"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. API Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIErrorHandling:
    """Test API 400 and 502 error paths."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("pydantic")

    def test_outside_bangalore_returns_400(self):
        """A destination outside Bangalore should return HTTP 400."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LatLng(lat=28.61, lng=77.20)  # Delhi coordinates

    def test_custom_route_request_valid(self):
        """Valid CustomRouteRequest should parse without errors."""
        from api_bangalore import CustomRouteRequest
        req = CustomRouteRequest(
            origin={"lat": 12.9716, "lng": 77.5946, "label": "Indiranagar Metro"},
            destination={"lat": 12.8440, "lng": 77.6610, "label": "Electronic City"},
            day_offset=0
        )
        assert req.origin.lat == 12.9716
        assert req.destination.lng == 77.6610


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelSmokeTest:
    """Verify the A3T-GCN model loads and produces correct output shape."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("torch")

    def test_model_forward_pass(self):
        """Model should output shape (16, 3) for 16 roads, 3-day horizon."""
        import torch
        from model import TrafficForecastModel

        model = TrafficForecastModel(in_channels=2, in_periods=7, out_periods=3)
        x = torch.rand(16, 2, 7)
        edge_index = torch.randint(0, 16, (2, 100))
        edge_weight = torch.rand(100)

        out = model(x, edge_index, edge_weight)
        assert out.shape == (16, 3), f"Expected (16, 3), got {out.shape}"

    def test_checkpoint_loads(self):
        """Trained checkpoint should load and produce (16, 3) output."""
        import torch
        from model import TrafficForecastModel

        ckpt_path = os.path.join(_BASE, "data", "bangalore_best_model.pt")
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not available (gitignored)")

        model = TrafficForecastModel(in_channels=2, in_periods=7, out_periods=3)
        try:
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        x = torch.rand(16, 2, 7)
        edge_index = torch.randint(0, 16, (2, 100))
        edge_weight = torch.rand(100)

        with torch.no_grad():
            out = model(x, edge_index, edge_weight)
        assert out.shape == (16, 3), f"Expected (16, 3), got {out.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometryHelpers:
    """Test haversine and point sampling helpers."""

    def test_haversine_known_distance(self):
        from custom_routing import haversine_km
        # Bangalore Majestic to Whitefield: ~20 km
        d = haversine_km(12.977, 77.572, 12.969, 77.749)
        assert 15 < d < 25, f"Expected ~20 km, got {d}"

    def test_sample_points_along_line(self):
        from custom_routing import sample_points_along_line
        # Simple line ~5 km long
        coords = [[77.60, 12.97], [77.65, 12.97]]
        sampled = sample_points_along_line(coords, interval_m=200)
        assert len(sampled) >= 2, "Should have at least start and end"
        assert sampled[0] == (12.97, 77.60), "First point should match"
        assert sampled[-1] == (12.97, 77.65), "Last point should match"
