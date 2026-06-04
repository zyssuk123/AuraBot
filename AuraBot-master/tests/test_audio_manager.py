import unittest

from face_id.audio_manager import AudioManager


class TestAudioManagerConfig(unittest.TestCase):
    def test_placeholder_gemini_key_is_rejected(self):
        self.assertTrue(AudioManager._is_placeholder_gemini_key("your_gemini_api_key_here"))
        self.assertTrue(AudioManager._is_placeholder_gemini_key("  YOUR_GEMINI_API_KEY_HERE  "))
        self.assertTrue(AudioManager._is_valid_gemini_key("AIzaSyValidKeyExample123"))

    def test_gemini_error_message_for_auth_failure(self):
        self.assertEqual(
            AudioManager._get_gemini_error_message(401),
            "La clé Gemini est invalide ou expirée. Vérifie ta clé dans le fichier .env.",
        )

    def test_gemini_error_message_for_rate_limit(self):
        self.assertEqual(
            AudioManager._get_gemini_error_message(429),
            "Le service Gemini est temporairement surchargé. Réessaie dans quelques instants.",
        )


    def test_conversation_window_can_be_activated(self):
        audio = AudioManager.__new__(AudioManager)
        audio._conversation_until = 0.0

        self.assertFalse(audio._is_conversation_active())
        audio._activate_conversation(seconds=1.0)
        self.assertTrue(audio._is_conversation_active())


if __name__ == "__main__":
    unittest.main()
