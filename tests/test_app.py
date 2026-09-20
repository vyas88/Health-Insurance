"""App tests never issue live API calls."""
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.ai_interpretation import AIInterpretationError


# AppTest executes Streamlit widget interactions without a live browser.
# These tests complement visual checks: they detect exceptions, missing
# results and stale state, but do not judge pixel layout or text contrast.
class AppTests(unittest.TestCase):
    # Force a loader error, then simulate a changed artifact fingerprint.
    # The app must show the error and clear old assessments/explanations.
    def test_artifact_error_and_session_invalidation(self):
        import streamlit as st
        # Clear the shared resource cache before patching the loader; otherwise
        # a previous successful cached result could bypass the intended failure.
        st.cache_resource.clear()
        with patch('src.inference.load_artifacts', side_effect=ValueError('Artifact mismatch. Run python -m src.main.')):
            app = AppTest.from_file('app.py', default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertIn('Artifact mismatch', app.error[0].value)
        st.cache_resource.clear()
        app = AppTest.from_file('app.py', default_timeout=20).run()
        app.button[0].click().run()
        app.session_state['artifact_fingerprint'] = 'old-run'
        app.session_state['ai_cache'] = {'old': 'stale explanation'}
        app.run()
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn('ai_cache', app.session_state)

    # Walk the main user journey with disabled, successful and failed AI states.
    # patch temporarily substitutes service functions and restores them afterward.
    def test_pages_inputs_and_ai_cache(self):
        with patch('src.ai_interpretation.get_ai_settings',return_value=(None,'configured-model')):
            app=AppTest.from_file('app.py',default_timeout=20).run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric),3)
            app.button[1].click().run()
            self.assertTrue(app.info)
            # Editing a field must immediately hide the previous cards even before
            # the next submission, so an old result cannot appear to describe new inputs.
            app.number_input[0].set_value(36).run()
            self.assertEqual(len(app.metric),0)
            app.number_input[0].set_value(-1).run()
            app.button[0].click().run()
            self.assertTrue(app.error)
            # Visit every informational page and check that saved assets/data render
            # without a Streamlit exception.
            for page in ['How the Model Works','Results & Model Comparison','Dataset & Methodology']:
                app.radio[0].set_value(page).run()
                self.assertFalse(app.exception)
        with patch('src.ai_interpretation.get_ai_settings',return_value=('mock-key','configured-model')), patch('src.ai_interpretation.create_client'), patch('src.ai_interpretation.generate_ai_interpretation',return_value='Mock AI text') as generate:
            app=AppTest.from_file('app.py',default_timeout=20).run()
            app.button[0].click().run(); app.button[1].click().run()
            self.assertFalse(app.exception)
            # A second click with the same successful context must reuse cached text
            # rather than create another service call.
            self.assertEqual(generate.call_count,1)
            app.button[1].click().run()
            self.assertEqual(generate.call_count,1)
            # A changed profile invalidates the displayed prose. A new submitted
            # assessment can request a distinct explanation keyed to its new context.
            app.number_input[0].set_value(37).run()
            self.assertFalse(any('Mock AI text' in m.value for m in app.markdown))
            app.button[0].click().run(); app.button[1].click().run()
            self.assertEqual(generate.call_count,2)
        with patch('src.ai_interpretation.get_ai_settings',return_value=('mock-key','configured-model')), patch('src.ai_interpretation.create_client'), patch('src.ai_interpretation.generate_ai_interpretation',side_effect=AIInterpretationError('Unavailable')):
            app=AppTest.from_file('app.py',default_timeout=20).run()
            app.button[0].click().run(); app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.info)


# Support both direct execution and python -m unittest discover -s tests.
if __name__ == '__main__':
    unittest.main()
