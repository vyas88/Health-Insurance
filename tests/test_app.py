"""App tests never issue live API calls."""
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.ai_interpretation import AIInterpretationError


class AppTests(unittest.TestCase):
    def test_artifact_error_and_session_invalidation(self):
        import streamlit as st
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

    def test_pages_inputs_and_ai_cache(self):
        with patch('src.ai_interpretation.get_ai_settings',return_value=(None,'configured-model')):
            app=AppTest.from_file('app.py',default_timeout=20).run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric),3)
            app.button[1].click().run()
            self.assertTrue(app.info)
            app.number_input[0].set_value(36).run()
            self.assertEqual(len(app.metric),0)
            app.number_input[0].set_value(-1).run()
            app.button[0].click().run()
            self.assertTrue(app.error)
            for page in ['How the Model Works','Results & Model Comparison','Dataset & Methodology']:
                app.radio[0].set_value(page).run()
                self.assertFalse(app.exception)
        with patch('src.ai_interpretation.get_ai_settings',return_value=('mock-key','configured-model')), patch('src.ai_interpretation.create_client'), patch('src.ai_interpretation.generate_ai_interpretation',return_value='Mock AI text') as generate:
            app=AppTest.from_file('app.py',default_timeout=20).run()
            app.button[0].click().run(); app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generate.call_count,1)
            app.button[1].click().run()
            self.assertEqual(generate.call_count,1)
            app.number_input[0].set_value(37).run()
            self.assertFalse(any('Mock AI text' in m.value for m in app.markdown))
            app.button[0].click().run(); app.button[1].click().run()
            self.assertEqual(generate.call_count,2)
        with patch('src.ai_interpretation.get_ai_settings',return_value=('mock-key','configured-model')), patch('src.ai_interpretation.create_client'), patch('src.ai_interpretation.generate_ai_interpretation',side_effect=AIInterpretationError('Unavailable')):
            app=AppTest.from_file('app.py',default_timeout=20).run()
            app.button[0].click().run(); app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.info)


if __name__ == '__main__':
    unittest.main()
