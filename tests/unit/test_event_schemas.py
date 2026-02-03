"""
Unit tests for EventData schema validation.

Tests cover:
- Basic validation of required event_id field
- Type coercion for event_id (string to float, int to float)
- Key normalization (PascalCase to lowercase)
- Default values for optional fields
- Extra fields preservation
- Error handling for invalid data
"""

import pytest
from pydantic import ValidationError

from dendrite.data.event_schemas import EventData


class TestEventDataBasicValidation:
    """Test basic validation and field requirements."""

    def test_valid_event(self):
        """Test validation with all required fields."""
        event = EventData(event_id=100.0, event_type='trial_start')
        assert event.event_id == 100.0
        assert event.event_type == 'trial_start'

    def test_missing_event_id_raises_error(self):
        """Test that missing event_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EventData(event_type='test')
        assert 'event_id' in str(exc_info.value)

    def test_missing_event_type_raises_error(self):
        """Test that missing event_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EventData(event_id=1.0)
        assert 'event_type' in str(exc_info.value)


class TestEventIdCoercion:
    """Test event_id type coercion."""

    def test_event_id_from_int(self):
        """Test event_id coercion from int to float."""
        event = EventData(event_id=42, event_type='test')
        assert event.event_id == 42.0
        assert isinstance(event.event_id, float)

    def test_event_id_from_string_numeric(self):
        """Test event_id coercion from numeric string."""
        event = EventData(event_id='123.5', event_type='test')
        assert event.event_id == 123.5

    def test_event_id_from_string_integer(self):
        """Test event_id coercion from integer string."""
        event = EventData(event_id='100', event_type='test')
        assert event.event_id == 100.0

    def test_event_id_invalid_string_raises_error(self):
        """Test that non-numeric string raises clear error."""
        with pytest.raises(ValidationError) as exc_info:
            EventData(event_id='not_a_number', event_type='test')
        assert 'event_id must be numeric' in str(exc_info.value)

    def test_event_id_none_raises_error(self):
        """Test that None event_id raises error."""
        with pytest.raises(ValidationError) as exc_info:
            EventData(event_id=None, event_type='test')
        assert 'event_id' in str(exc_info.value)

    def test_event_id_empty_string_raises_error(self):
        """Test that empty string raises error."""
        with pytest.raises(ValidationError) as exc_info:
            EventData(event_id='', event_type='test')
        assert 'event_id must be numeric' in str(exc_info.value)


class TestKeyNormalization:
    """Test PascalCase to lowercase key normalization."""

    def test_pascal_case_event_id(self):
        """Test Event_ID key is normalized to event_id."""
        event = EventData.model_validate({'Event_ID': 1.0, 'Event_Type': 'test'})
        assert event.event_id == 1.0

    def test_pascal_case_event_type(self):
        """Test Event_Type key is normalized to event_type."""
        event = EventData.model_validate({'event_id': 1.0, 'Event_Type': 'test'})
        assert event.event_type == 'test'

    def test_upper_case_keys(self):
        """Test UPPERCASE keys are normalized."""
        event = EventData.model_validate({'EVENT_ID': 1.0, 'EVENT_TYPE': 'test'})
        assert event.event_id == 1.0
        assert event.event_type == 'test'

    def test_mixed_case_keys(self):
        """Test mixed case keys are normalized."""
        event = EventData.model_validate({
            'EvEnT_iD': 2.0,
            'eVeNt_TyPe': 'mixed'
        })
        assert event.event_id == 2.0
        assert event.event_type == 'mixed'


class TestExtraFields:
    """Test handling of extra (custom) fields."""

    def test_extra_fields_preserved(self):
        """Test that extra fields are preserved in model."""
        event = EventData.model_validate({
            'event_id': 1.0,
            'event_type': 'trial_start',
            'custom_field': 'custom_value',
            'another_field': 42
        })
        assert event.event_id == 1.0
        assert event.event_type == 'trial_start'
        assert event.custom_field == 'custom_value'
        assert event.another_field == 42

    def test_extra_fields_in_dump(self):
        """Test that extra fields appear in model_dump output."""
        event = EventData.model_validate({
            'event_id': 1.0,
            'event_type': 'test',
            'latency_ms_raw': 15.5,
            'task_name': 'motor_imagery'
        })
        dumped = event.model_dump()
        assert dumped['event_id'] == 1.0
        assert dumped['latency_ms_raw'] == 15.5
        assert dumped['task_name'] == 'motor_imagery'

    def test_extra_fields_key_normalization(self):
        """Test that extra field keys are also normalized to lowercase."""
        event = EventData.model_validate({
            'event_id': 1.0,
            'event_type': 'test',
            'Latency_MS_Raw': 10.0,
            'Task_Name': 'test'
        })
        dumped = event.model_dump()
        assert 'latency_ms_raw' in dumped
        assert 'task_name' in dumped


class TestModelDumpOutput:
    """Test model_dump output for use in event processing."""

    def test_model_dump_contains_all_fields(self):
        """Test that model_dump returns complete event data."""
        event = EventData.model_validate({
            'event_id': 100,
            'event_type': 'trial_end',
            'extra_data': 'value'
        })
        dumped = event.model_dump()
        assert dumped == {
            'event_id': 100.0,
            'event_type': 'trial_end',
            'extra_data': 'value'
        }

    def test_model_dump_required_fields_only(self):
        """Test model_dump with only required fields."""
        event = EventData(event_id=1.0, event_type='test')
        dumped = event.model_dump()
        assert dumped == {'event_id': 1.0, 'event_type': 'test'}


class TestRealWorldEventFormats:
    """Test validation with real-world event payload formats."""

    def test_lsl_event_payload(self):
        """Test typical LSL event payload format."""
        payload = {
            'event_id': 1,
            'event_type': 'trial_start',
            'task': 'motor_imagery',
            'class': 'left_hand'
        }
        event = EventData.model_validate(payload)
        assert event.event_id == 1.0
        assert event.event_type == 'trial_start'

    def test_legacy_pascal_case_payload(self):
        """Test legacy PascalCase payload format."""
        payload = {
            'Event_ID': 2,
            'Event_Type': 'trial_end',
            'Accuracy': 0.85
        }
        event = EventData.model_validate(payload)
        assert event.event_id == 2.0
        assert event.event_type == 'trial_end'
        dumped = event.model_dump()
        assert dumped['accuracy'] == 0.85

    def test_latency_update_event(self):
        """Test latency_update event format used for E2E tracking."""
        payload = {
            'event_id': 999,
            'event_type': 'latency_update',
            'latency_ms_raw': 12.5
        }
        event = EventData.model_validate(payload)
        assert event.event_id == 999.0
        assert event.event_type == 'latency_update'
        assert event.model_dump()['latency_ms_raw'] == 12.5

    def test_event_with_nested_data(self):
        """Test event with nested dict as extra field."""
        payload = {
            'event_id': 1,
            'event_type': 'metadata',
            'params': {'threshold': 0.5, 'enabled': True}
        }
        event = EventData.model_validate(payload)
        dumped = event.model_dump()
        assert dumped['params'] == {'threshold': 0.5, 'enabled': True}
