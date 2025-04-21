import pyttsx3
import os
import random

print(f"Current working directory: {os.getcwd()}")

def speak_pyttsx3(text, rate=150, volume=1.0):
    """Converts text to speech using pyttsx3 and plays it."""
    try:
        engine = pyttsx3.init()

        # Set speech rate
        engine.setProperty('rate', rate)

        # Set volume (0.0 to 1.0)
        engine.setProperty('volume', volume)

        engine.say(text)
        engine.runAndWait()
        engine.stop()

    except Exception as e:
        print(f"Error during pyttsx3: {e}")

if __name__ == "__main__":
    object="object"
    distance = random.randint(1,100)
    test_phrases = f"This is a test phrase . The {object} is {distance} cm away"

    # random_phrase = random.choice(test_phrases)

    # Ensure your Bluetooth earphones are connected and set as the default audio output
    print("Make sure your Bluetooth earphones are connected and set as the default audio output.")
    print(f"Speaking: '{test_phrases}'")

    speak_pyttsx3(test_phrases)
    print("Test spoken (hopefully on your Bluetooth earphones).")
