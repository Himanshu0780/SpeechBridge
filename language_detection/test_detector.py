"""
language_detection/test_detector.py
-------------------------------------
Quick standalone test for the language detector.

Run: python -m language_detection.test_detector
"""

def test_detector():
    from language_detection.detector import LanguageDetector

    detector = LanguageDetector()

    test_cases = [
        ("नमस्ते दुनिया, कैसे हो आप?",             "Hindi"),
        ("வணக்கம்! நான் நலமாக இருக்கிறேன்.",         "Tamil"),
        ("Hello, how are you today?",               "English"),
        ("আমি বাংলা ভাষায় কথা বলছি।",              "Bengali"),
        ("నమస్కారం, మీరు ఎలా ఉన్నారు?",             "Telugu"),
        ("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",           "Punjabi"),
        ("नमस्कार, तुम्ही कसे आहात?",                "Marathi"),
        ("નમસ્તે, તમે કેમ છો?",                     "Gujarati"),
        ("ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",               "Kannada"),
        ("നമസ്കാരം, നിങ്ങൾ എങ്ങനെ ഉണ്ട്?",          "Malayalam"),
    ]

    print("\n🌐 Language Detection Test Results\n" + "─" * 50)
    passed = 0
    for text, expected in test_cases:
        result = detector.detect(text)
        detected = result["language"]
        confidence = result["confidence"]
        ok = detected == expected
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(f"  {status} Expected: {expected:12} | Got: {detected:12} | Conf: {confidence:.2f}")
        print(f"    Text: {text[:50]}")

    print(f"\n  Passed: {passed}/{len(test_cases)}\n")


if __name__ == "__main__":
    test_detector()
