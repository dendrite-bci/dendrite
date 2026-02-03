"""
Unit tests for SharedState cross-process shared memory.

Tests float/dict operations, cross-process communication, pickling,
wait_for timeout behavior, and helper methods.
"""

import time
import multiprocessing
import pytest

from dendrite.utils import SharedState


class TestSharedStateFloatOperations:
    """Test suite for float value operations."""

    def test_set_and_get_float(self):
        """Test setting and retrieving float values."""
        state = SharedState()
        try:
            state.set('test_metric', 42.5)
            result = state.get('test_metric')
            assert result == 42.5
        finally:
            state.cleanup()

    def test_get_nonexistent_returns_none(self):
        """Test that getting a nonexistent key returns None."""
        state = SharedState()
        try:
            result = state.get('nonexistent_key')
            assert result is None
        finally:
            state.cleanup()

    def test_int_stored_correctly(self):
        """Test that integers are stored and retrieved correctly."""
        state = SharedState()
        try:
            state.set('int_value', 100)
            result = state.get('int_value')
            assert result == 100
        finally:
            state.cleanup()

    def test_overwrite_float_value(self):
        """Test that float values can be overwritten."""
        state = SharedState()
        try:
            state.set('metric', 1.0)
            state.set('metric', 2.0)
            assert state.get('metric') == 2.0
        finally:
            state.cleanup()


class TestSharedStateDictOperations:
    """Test suite for dict value operations."""

    def test_set_and_get_dict(self):
        """Test setting and retrieving dict values."""
        state = SharedState()
        try:
            test_dict = {'labels': ['Fp1', 'Fp2'], 'count': 2}
            state.set('config', test_dict)
            result = state.get('config')
            assert result == test_dict
        finally:
            state.cleanup()

    def test_nested_dict_roundtrip(self):
        """Test complex nested structures survive roundtrip."""
        state = SharedState()
        try:
            complex_dict = {
                'level1': {
                    'level2': {
                        'values': [1, 2, 3],
                        'nested_list': [[1, 2], [3, 4]]
                    }
                },
                'array': [1.5, 2.5, 3.5]
            }
            state.set('complex', complex_dict)
            result = state.get('complex')
            assert result == complex_dict
        finally:
            state.cleanup()

    def test_list_storage(self):
        """Test that lists can be stored and retrieved."""
        state = SharedState()
        try:
            test_list = [1, 2, 3, {'nested': 'dict'}]
            state.set('my_list', test_list)
            result = state.get('my_list')
            assert result == test_list
        finally:
            state.cleanup()


class TestSharedStateCrossProcess:
    """Test suite for cross-process communication."""

    @pytest.mark.slow
    def test_cross_process_float(self):
        """Test float set in child process, read in parent."""
        state = SharedState()
        try:
            # Pre-set a value to ensure key exists
            state.set('child_metric', 0.0)

            def child_set_float(shared_state):
                shared_state.set('child_metric', 99.9)

            proc = multiprocessing.Process(target=child_set_float, args=(state,))
            proc.start()
            proc.join(timeout=5)

            result = state.get('child_metric')
            assert result == 99.9
        finally:
            state.cleanup()

    @pytest.mark.slow
    def test_cross_process_dict(self):
        """Test dict set in child process, read in parent."""
        state = SharedState()
        try:
            def child_set_dict(shared_state):
                shared_state.set('channel_info', {
                    'labels': ['Ch1', 'Ch2'],
                    'total': 2
                })

            proc = multiprocessing.Process(target=child_set_dict, args=(state,))
            proc.start()
            proc.join(timeout=5)

            result = state.get('channel_info')
            assert result == {'labels': ['Ch1', 'Ch2'], 'total': 2}
        finally:
            state.cleanup()


class TestSharedStatePickling:
    """Test suite for process passing (spawn context compatibility).

    Note: Manager-based SharedState can't be directly pickled with pickle.dumps()
    due to security restrictions on AuthenticationString. However, it works
    correctly when passed to child processes via multiprocessing.Process.
    """

    @pytest.mark.slow
    def test_data_accessible_in_child_process(self):
        """Test that data set in parent is accessible in child process."""
        state = SharedState()
        try:
            # Set values in parent
            state.set('float_value', 123.0)
            state.set('dict_value', {'key': 'value'})

            results = multiprocessing.Queue()

            def child_read(shared_state, queue):
                queue.put({
                    'float': shared_state.get('float_value'),
                    'dict': shared_state.get('dict_value')
                })

            proc = multiprocessing.Process(target=child_read, args=(state, results))
            proc.start()
            proc.join(timeout=5)

            result = results.get(timeout=1)
            assert result['float'] == 123.0
            assert result['dict'] == {'key': 'value'}
        finally:
            state.cleanup()

    @pytest.mark.slow
    def test_child_can_set_values(self):
        """Test that child process can set new values."""
        state = SharedState()
        try:
            def child_set(shared_state):
                shared_state.set('child_float', 456.0)
                shared_state.set('child_dict', {'from': 'child'})

            proc = multiprocessing.Process(target=child_set, args=(state,))
            proc.start()
            proc.join(timeout=5)

            assert state.get('child_float') == 456.0
            assert state.get('child_dict') == {'from': 'child'}
        finally:
            state.cleanup()


class TestSharedStateWaitFor:
    """Test suite for wait_for() blocking method."""

    @pytest.mark.slow
    def test_wait_for_returns_when_set(self):
        """Test wait_for returns value once set by another process."""
        state = SharedState()
        try:
            def delayed_set(shared_state):
                time.sleep(0.2)
                shared_state.set('channel_info', {'ready': True})

            proc = multiprocessing.Process(target=delayed_set, args=(state,))
            proc.start()

            start = time.monotonic()
            result = state.wait_for('channel_info', timeout=5.0, poll_interval=0.05)
            elapsed = time.monotonic() - start

            proc.join(timeout=2)

            assert result == {'ready': True}
            assert 0.1 < elapsed < 2.0  # Should return quickly after set
        finally:
            state.cleanup()

    def test_wait_for_timeout_returns_none(self):
        """Test wait_for returns None when timeout exceeded."""
        state = SharedState()
        try:
            # Don't set the value - let it timeout
            start = time.monotonic()
            result = state.wait_for('never_set_key', timeout=0.3, poll_interval=0.05)
            elapsed = time.monotonic() - start

            assert result is None
            assert 0.25 < elapsed < 0.5  # Should timeout around 0.3s
        finally:
            state.cleanup()


class TestSharedStateHelperMethods:
    """Test suite for helper methods."""

    def test_cleanup_clears_data(self):
        """Test cleanup properly clears the shared data."""
        state = SharedState()
        state.set('test_value', 42.0)

        # Verify value exists
        assert state.get('test_value') == 42.0

        # Cleanup should clear
        state.cleanup()

        # Note: After cleanup, the manager is shut down so we can't access data
        # This test verifies cleanup doesn't raise exceptions

    def test_clear_removes_key(self):
        """Test clear() removes a specific key."""
        state = SharedState()
        try:
            # Set a value
            state.set('channel_info', {'key': 'value', 'nested': {'a': 1}})
            assert state.get('channel_info') == {'key': 'value', 'nested': {'a': 1}}

            # Clear should remove it
            state.clear('channel_info')
            assert state.get('channel_info') is None

            # Can set again after clear
            state.set('channel_info', {'new': 'data'})
            assert state.get('channel_info') == {'new': 'data'}
        finally:
            state.cleanup()

    def test_clear_nonexistent_key_is_noop(self):
        """Test clear() on non-existent key does nothing."""
        state = SharedState()
        try:
            # Should not raise
            state.clear('nonexistent_key')
        finally:
            state.cleanup()

    def test_get_with_default(self):
        """Test get() returns default for missing keys."""
        state = SharedState()
        try:
            result = state.get('missing_key', default='fallback')
            assert result == 'fallback'
        finally:
            state.cleanup()


class TestSharedStatePlatformCompatibility:
    """Test platform-specific behaviors and resource management."""

    def test_context_manager_cleanup(self):
        """Test context manager properly cleans up resources."""
        with SharedState() as state:
            state.set('test', 42.0)
            assert state.get('test') == 42.0

        # After context exit, cleanup should have been called
        # (Manager is shut down, but this shouldn't raise)

    def test_context_manager_cleanup_on_exception(self):
        """Test context manager cleans up even when exception raised."""
        try:
            with SharedState() as state:
                state.set('test', 42.0)
                raise ValueError("Intentional test exception")
        except ValueError:
            pass

        # Context manager should have called cleanup despite exception

    @pytest.mark.slow
    def test_concurrent_writes_no_corruption(self):
        """Test multiple processes writing simultaneously don't corrupt data."""
        state = SharedState()
        try:
            # Pre-set the key
            state.set('counter', 0.0)

            def writer(shared_state, value):
                for _ in range(100):
                    shared_state.set('counter', value)

            procs = [
                multiprocessing.Process(target=writer, args=(state, float(i)))
                for i in range(4)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=10)

            # Value should be one of 0.0, 1.0, 2.0, or 3.0 (not corrupted)
            result = state.get('counter')
            assert result in [0.0, 1.0, 2.0, 3.0]
        finally:
            state.cleanup()

