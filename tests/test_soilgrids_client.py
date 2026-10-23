import pytest
from src.data_collection.soilgrids_client import (
    validate_coordinates, scale_value, InvalidCoordinateError)


def test_valid():
    assert validate_coordinates(11.94, 79.49) == (11.94, 79.49)


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, 181), ("a", 1), (float("nan"), 0)])
def test_invalid(lat, lon):
    with pytest.raises(InvalidCoordinateError):
        validate_coordinates(lat, lon)


def test_scaling_and_nodata():
    assert scale_value(65, -32768, 10) == 6.5
    assert scale_value(-32768, -32768, 10) is None   # missing, never 0
