// Example prompt library. SFX prompts are hard to invent on the spot, so
// the page offers these as clickable cards: a click fills the focused
// prompt field and never renders by itself. The first entry of each
// category marked `probe` is one of the capability-probe prompts measured
// in notes (results_sfx.md), so their behaviour is known.
export const LIBRARY = [
  {
    category: "Ambience",
    items: [
      { text: "Steady heavy rain on a tin roof", probe: true },
      { text: "Crowd murmur in a large indoor hall", probe: true },
      { text: "Busy city street traffic ambience, distant horns" },
      { text: "Forest at dawn, birdsong and light wind" },
      { text: "Ocean waves on a pebble beach" },
      { text: "Cafe ambience, cups clinking, quiet chatter" },
      { text: "Spaceship interior hum, low drone and soft beeps" },
    ],
  },
  {
    category: "Weather",
    items: [
      { text: "Thunderstorm with rolling thunder and heavy rain" },
      { text: "Wind howling through a canyon" },
      { text: "Light drizzle on leaves, gentle and calm" },
      { text: "Hail storm hitting a car roof" },
      { text: "Blizzard, harsh wind and blowing snow" },
    ],
  },
  {
    category: "Machines",
    items: [
      { text: "Idling diesel engine, constant rumble", probe: true },
      { text: "Factory machinery, rhythmic industrial clanking" },
      { text: "Old steam train passing, chugging and whistle" },
      { text: "Electric drill boring into wood" },
      { text: "Server room hum, cooling fans" },
    ],
  },
  {
    category: "Foley",
    items: [
      { text: "Footsteps on gravel, a few steps", probe: true },
      { text: "Glass bottle shattering on concrete", probe: true },
      { text: "Wooden door creak, slow opening" },
      { text: "Paper crumpling in hands" },
      { text: "Keys jingling, dropped on a table" },
      { text: "Crackling campfire at night, crickets" },
    ],
  },
  {
    category: "Impacts",
    items: [
      { text: "Heavy metal door slam, single impact, reverberant warehouse", probe: true },
      { text: "Deep cinematic boom impact with long sub tail", probe: true },
      { text: "Punch hitting a leather bag" },
      { text: "Single gunshot outdoors with echo" },
      { text: "Large rock falling onto dirt" },
    ],
  },
  {
    category: "UI",
    items: [
      { text: "Short UI confirm chime, clean bright notification", probe: true },
      { text: "Soft UI click, minimal interface tap", probe: true },
      { text: "Error buzz, short low warning tone" },
      { text: "Coin pickup, retro game sound" },
      { text: "Message pop, playful bubble sound" },
    ],
  },
  {
    category: "Creatures",
    items: [
      { text: "Dog barking next to a waterfall", probe: true },
      { text: "Wolf howling at night, distant" },
      { text: "Large monster growl, deep and wet" },
      { text: "Swarm of bees buzzing close" },
      { text: "Seagulls calling over a harbor" },
    ],
  },
  {
    category: "Risers and whooshes",
    items: [
      { text: "Rising tension riser, building synth swell", probe: true },
      { text: "Futuristic laser blast, sharp energy pulse, arcade style", probe: true },
      { text: "Whoosh swipe, fast air movement past the microphone" },
      { text: "Magical sparkle shimmer, fantasy spell burst" },
      { text: "Reverse cymbal swell into a hit" },
    ],
  },
];
