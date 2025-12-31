# CONFIGURATION & LORE
# Customize the horror elements here.

# The "Normal" Facade
LOCATIONS = [
    "Sector 4", "North District", "The Valley", "Greater Suburbia", "Zone 0"
]

WEATHER_CONDITIONS = [
    "Clear Skies", "Heavy Static", "Memory Fog", "Data Rain", "Void Hail", "Signal Decay"
]

# The "Entity" Invasion
ENTITIES = [
    "THE_LONE_SHOOTER", # As requested
    "SUBJECT_6",
    "THE_BROADCAST_ITSELF",
    "UNREGISTERED_HYPEROBJECT"
]

# Text to scrape/inject into the video to make it unsettling
# These are fallback phrases if live scraping fails
CREEPY_PHRASES = [
    "IT IS IN THE WALLS",
    "DO NOT ANSWER THE PHONE",
    "LOOK BEHIND YOU",
    "THE SKY IS FALSE",
    "MEMORY DELETED"
]

# ARG Settings
# Viewers must decode this Base64 string to get the "Truth"
HIDDEN_TRUTH = "THEY_ARE_REPLACING_US_ONE_BY_ONE"

# Video Settings
VIDEO_RES = (1280, 720)
FPS = 24
FONT_PATH = "assets/vcr.ttf"  # User needs to drop a font here, or it defaults

