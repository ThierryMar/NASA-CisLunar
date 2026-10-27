"use strict";

// Le menu reste visible sans JavaScript pour garder les liens accessibles.
const header = document.querySelector(".header");
const menuButton = document.querySelector(".menu-toggle");
const navigation = document.querySelector("#navigation");
header.classList.add("enhanced");

function closeMenu() {
  navigation.classList.remove("is-open");
  menuButton.setAttribute("aria-expanded", "false");
}

menuButton.addEventListener("click", () => {
  const open = menuButton.getAttribute("aria-expanded") !== "true";
  navigation.classList.toggle("is-open", open);
  menuButton.setAttribute("aria-expanded", String(open));
});
navigation.addEventListener("click", (event) => {
  if (event.target.closest("a")) closeMenu();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
    closeMenu();
    menuButton.focus();
  }
});

// Animation pédagogique : aucune donnée astronomique ou topographique.
const scenario = document.querySelector("#scenario");
const scenarioValue = document.querySelector("#scenario-value");
const sun = document.querySelector("#sun-object");
const earth = document.querySelector("#earth-object");
const sunStatus = document.querySelector("#sun-status");
const earthStatus = document.querySelector("#earth-status");
const horizonY = 240;

function updateDiagram() {
  const progress = Number(scenario.value) / 100;
  const sunX = 160 + progress * 380;
  const sunY = 285 - Math.sin(progress * Math.PI) * 173;
  const earthX = 620 + progress * 160;
  const earthY = 180 + Math.sin((progress - 0.5) * Math.PI) * 110;
  sun.setAttribute("transform", `translate(${sunX} ${sunY})`);
  earth.setAttribute("transform", `translate(${earthX} ${earthY})`);
  scenarioValue.value = `${scenario.value} / 100`;
  const visibility = (y) => y < horizonY ? "Au-dessus de l’horizon" : "Sous l’horizon";
  sunStatus.textContent = visibility(sunY);
  earthStatus.textContent = visibility(earthY);
}

scenario.addEventListener("input", updateDiagram);
updateDiagram();
