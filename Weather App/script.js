const apiKey = "37338b0131843ad93b038ce709d2c989";

window.onload = () => {
    document.body.classList.add("default");
};

async function getWeather() {
    const city = document.getElementById("city").value.trim();
    if (!city) {
        alert("Please enter a city name!");
        return;
    }

    document.body.classList.remove("default");
    const intro = document.getElementById("intro");
    if (intro) intro.style.display = "none";

    const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        const weatherInfo = document.getElementById("weather-info");
        weatherInfo.innerHTML = "";

        if (data.cod === 200) {
            const temp = data.main.temp;
            const humidity = data.main.humidity;
            const windSpeed = data.wind.speed;
            const weatherCondition = data.weather[0].main;
            const icon = data.weather[0].icon;

            weatherInfo.innerHTML = `
                <h2>${data.name}, ${data.sys.country}</h2>
                <p>Temperature: ${temp}°C</p>
                <p>Humidity: ${humidity}%</p>
                <p>Wind Speed: ${windSpeed} m/s</p>
                <p>Condition: ${weatherCondition}</p>
                <img src="https://openweathermap.org/img/wn/${icon}@2x.png" alt="Weather Icon">
            `;

            const isNight = icon.includes("n");
            setWeatherBackground(weatherCondition, isNight);
        } else {
            weatherInfo.innerHTML = `<p>${data.message}</p>`;
        }
    } catch (error) {
        console.error("Error fetching data:", error);
        document.getElementById("weather-info").innerHTML = `<p>Failed to fetch weather data.</p>`;
    }
}

function setWeatherBackground(condition, isNight) {
    const body = document.body;
    body.className = '';
    removeVisuals();

    const conditionKey = `${condition.toLowerCase()}-${isNight ? "night" : "day"}`;

    const classMap = {
        "clear-day": "sunny",
        "clear-night": "night",
        "clouds-day": "cloudy-day",
        "clouds-night": "cloudy-night",
        "rain-day": "rainy-day",
        "rain-night": "rainy-night",
        "thunderstorm-day": "thunder-day",
        "thunderstorm-night": "thunder-night",
        "snow-day": "snow-day",
        "snow-night": "snow-night",
        "mist-day": "misty",
        "mist-night": "misty",
        "fog-day": "misty",
        "fog-night": "misty",
        "haze-day": "hazy",
        "haze-night": "hazy",
        "drizzle-day": "drizzle",
        "drizzle-night": "drizzle"
    };

    const className = classMap[conditionKey] || 'default';
    body.classList.add(className);
    body.classList.add("fade-in");
    setTimeout(() => body.classList.remove("fade-in"), 1000);

    const visuals = document.getElementById("visuals");

    if (isNight && condition.toLowerCase() === "clear") {
        addMoon();
        addStars();
    } else if (!isNight && condition.toLowerCase() === "clear") {
        addSun();
        generateClearDayClouds();
    }

    if (condition.toLowerCase().includes("thunderstorm")) {
        body.classList.add("flash");
    }
}

function removeVisuals() {
    document.querySelectorAll(".moon, .sun, .cloud, .stars").forEach(el => el.remove());
    document.body.classList.remove("flash");
}

function addSun() {
    const sun = document.createElement('div');
    sun.classList.add('sun');
    document.getElementById("visuals").appendChild(sun);
}

function addMoon() {
    const moon = document.createElement('div');
    moon.classList.add('moon');
    document.getElementById("visuals").appendChild(moon);
}

function addStars() {
    const stars = document.createElement('div');
    stars.classList.add('stars');
    document.getElementById("visuals").appendChild(stars);
}

function createCloud(size, delay) {
    const cloud = document.createElement('div');
    cloud.className = `cloud ${size}`;
    cloud.style.animationDelay = `${delay}s`;
    cloud.style.zIndex = '0';
    document.getElementById("visuals").appendChild(cloud);
}

function generateClearDayClouds() {
    // Remove old clouds if any
    document.querySelectorAll('.cloud').forEach(el => el.remove());

    // Create floating clouds
    createCloud('small', 0);
    createCloud('medium', 15);
    createCloud('large', 30);
}

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("weather-form");
    if (form) {
        form.addEventListener("submit", function (e) {
            e.preventDefault();
            getWeather();
        });
    }
});
