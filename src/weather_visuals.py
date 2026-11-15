"""Small animated, condition-aware weather scenes for TrafficSense."""
import html

WEATHER_SCENES = {
    "rain": ("🌧️", "Rainy conditions", "A person with an umbrella walks through animated rain.", "rain"),
    "drizzle": ("🌦️", "Light rain", "Light animated raindrops with a person carrying an umbrella.", "rain"),
    "thunder": ("⛈️", "Thunderstorm", "Storm clouds, rain and a brief lightning flash.", "storm"),
    "snow": ("❄️", "Snowy conditions", "Animated snowflakes drift across the scene.", "snow"),
    "cloud": ("☁️", "Cloudy conditions", "Soft clouds drift slowly across the sky.", "cloud"),
    "overcast": ("☁️", "Overcast conditions", "A calm, overcast sky with drifting clouds.", "cloud"),
    "mist": ("🌫️", "Misty conditions", "Layers of mist move gently across the road.", "haze"),
    "haze": ("🌫️", "Hazy conditions", "Soft haze moves across the distant skyline.", "haze"),
    "smoke": ("🌫️", "Smoky conditions", "Smoke drifts across the road and skyline.", "haze"),
    "fog": ("🌫️", "Foggy conditions", "Low fog drifts across the road.", "haze"),
    "clear": ("☀️", "Clear conditions", "A soft sun glow over a clear road scene.", "clear"),
    "sun": ("☀️", "Clear conditions", "A soft sun glow over a clear road scene.", "clear"),
}

CSS = """
<style>
.ts-weather-scene{position:relative;overflow:hidden;min-height:148px;border:1px solid #e4ded2;border-radius:16px;
background:linear-gradient(180deg,#e8eff0 0%,#f6f1e7 72%);padding:14px 16px;color:#263238;margin:.35rem 0 1rem}
.ts-weather-title{font:800 .72rem 'DM Sans',sans-serif;letter-spacing:.13em;text-transform:uppercase;color:#68766f}
.ts-weather-main{font:800 1.12rem 'Manrope',sans-serif;color:#263238;margin-top:4px}
.ts-weather-copy{font:.82rem 'DM Sans',sans-serif;color:#65736e;max-width:75%;margin-top:4px}
.ts-weather-icon{position:absolute;right:22px;top:18px;font-size:2.6rem;z-index:2;animation:ts-float 3s ease-in-out infinite}
.ts-road{position:absolute;left:0;right:0;bottom:0;height:27px;background:#41494b}
.ts-road:after{content:'';position:absolute;left:0;right:0;top:12px;border-top:2px dashed #e9dfc9;opacity:.9}
.ts-person{position:absolute;right:95px;bottom:19px;font-size:1.6rem;z-index:3;animation:ts-walk 1.8s ease-in-out infinite}
.ts-weather-scene.rain:before,.ts-weather-scene.storm:before{content:' ';position:absolute;inset:0;background:repeating-linear-gradient(105deg,transparent 0 18px,rgba(79,135,158,.45) 19px 21px,transparent 22px 38px);animation:ts-rain .55s linear infinite}
.ts-weather-scene.snow:before{content:'❄   ·    ❄       ·  ❄';position:absolute;inset:35px 10px auto;color:#fff;font-size:1.25rem;letter-spacing:15px;animation:ts-snow 4s linear infinite}
.ts-weather-scene.cloud .ts-weather-icon{animation:ts-cloud 7s ease-in-out infinite}
.ts-weather-scene.haze:before{content:'';position:absolute;left:-10%;right:-10%;bottom:28px;height:50px;background:repeating-linear-gradient(180deg,rgba(190,195,187,.0) 0 8px,rgba(178,187,180,.42) 9px 17px);filter:blur(4px);animation:ts-haze 5s ease-in-out infinite}
.ts-weather-scene.clear .ts-weather-icon{animation:ts-sun 4s ease-in-out infinite}
@keyframes ts-float{50%{transform:translateY(-6px)}} @keyframes ts-walk{0%,100%{transform:translateX(0)}50%{transform:translateX(-7px)}}
@keyframes ts-rain{to{background-position:0 40px}} @keyframes ts-snow{0%{transform:translateY(-5px)}100%{transform:translateY(75px)}}
@keyframes ts-cloud{50%{transform:translateX(-12px)}} @keyframes ts-haze{50%{transform:translateX(12px);opacity:.6}} @keyframes ts-sun{50%{transform:scale(1.1)}}
.ts-traffic-result{border-radius:16px;padding:16px 18px;border:1px solid;margin:.5rem 0 1rem}
.ts-traffic-result .eyebrow{font:800 .68rem 'DM Sans',sans-serif;letter-spacing:.14em;text-transform:uppercase;opacity:.78}
.ts-traffic-result .value{font:800 1.5rem 'Manrope',sans-serif;margin-top:3px}
.ts-traffic-result .detail{font:.85rem 'DM Sans',sans-serif;margin-top:3px;opacity:.88}
.ts-low{background:#e7f1e8;border-color:#bfd5c2;color:#35664a}.ts-moderate{background:#fbf1dc;border-color:#e8d3a5;color:#8a641e}.ts-high{background:#f6e6e2;border-color:#e4c0b8;color:#a34f46}
</style>
"""

def weather_scene(condition: str, show_person: bool = True) -> str:
    value = str(condition or "").strip().lower()

    # Match specific conditions before broader conditions.
    aliases = [
        (("thunderstorm", "thunder", "storm"), "thunder"),
        (("drizzle",), "drizzle"),
        (("rain", "rainy", "shower"), "rain"),
        (("snow", "sleet", "blizzard"), "snow"),
        (("smoke", "smoky"), "smoke"),
        (("mist",), "mist"),
        (("fog",), "fog"),
        (("haze",), "haze"),
        (("overcast",), "overcast"),
        (("cloud", "cloudy"), "cloud"),
        (("clear", "sunny", "sun"), "clear"),
    ]

    key = None
    for terms, scene_key in aliases:
        if any(term in value for term in terms):
            key = scene_key
            break

    # Unknown conditions should never incorrectly display clear weather.
    if key is None:
        icon = "🌫️"
        title = "Other weather conditions"
        copy = f"Weather condition: {value or 'Not specified'}."
        scene_class = "haze"
    else:
        icon, title, copy, scene_class = WEATHER_SCENES[key]

    person = (
        '<div class="ts-person">🚶🏻‍♀️</div>'
        if show_person and scene_class in {"rain", "storm", "haze"}
        else ""
    )

    return (
        CSS
        + f'<div class="ts-weather-scene {scene_class}">'
        + '<div class="ts-weather-title">LIVE SCENARIO WEATHER</div>'
        + f'<div class="ts-weather-main">{html.escape(title)}</div>'
        + f'<div class="ts-weather-copy">{html.escape(copy)}</div>'
        + f'<div class="ts-weather-icon">{icon}</div>'
        + person
        + '<div class="ts-road"></div></div>'
    )


def traffic_result(category: str, prediction: float, low_max: float, moderate_max: float) -> str:
    styles = {"Low": ("ts-low", "LOW TRAFFIC · GOOD TRAVEL CONDITIONS"),
              "Moderate": ("ts-moderate", "MODERATE TRAFFIC · ALLOW EXTRA TIME"),
              "High": ("ts-high", "HIGH TRAFFIC · EXPECT CONGESTION")}
    css_class, label = styles.get(category, styles["Moderate"])
    return (CSS + f'<div class="ts-traffic-result {css_class}">'
            f'<div class="eyebrow">{label}</div><div class="value">{prediction:,.0f} vehicles / hour</div>'
            f'<div class="detail">Traffic category: {html.escape(category)} · Based on the model\'s learned traffic thresholds.</div></div>')
