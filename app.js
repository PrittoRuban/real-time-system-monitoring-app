// Initialize charts using Chart.js
const ctxCpu = document.getElementById("cpuChart").getContext("2d");
const ctxMemory = document.getElementById("memoryChart").getContext("2d");
const ctxDisk = document.getElementById("diskChart").getContext("2d");
const ctxBattery = document.getElementById("batteryChart").getContext("2d");
const ctxGpu = document.getElementById("gpuChart").getContext("2d"); 

let cpuChart = new Chart(ctxCpu, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "CPU Usage (%)",
        data: [],
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  },
  options: { scales: { y: { beginAtZero: true, max: 100 } } },
});

let memoryChart = new Chart(ctxMemory, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Memory Usage (%)",
        data: [],
        borderColor: "rgba(153, 102, 255, 1)",
        backgroundColor: "rgba(153, 102, 255, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  },
  options: { scales: { y: { beginAtZero: true, max: 100 } } },
});

let diskChart = new Chart(ctxDisk, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Disk Usage (%)",
        data: [],
        borderColor: "rgba(255, 159, 64, 1)",
        backgroundColor: "rgba(255, 159, 64, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  },
  options: { scales: { y: { beginAtZero: true, max: 100 } } },
});

let batteryChart = new Chart(ctxBattery, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Battery (%)",
        data: [],
        borderColor: "rgba(255, 99, 132, 1)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  },
  options: { scales: { y: { beginAtZero: true, max: 100 } } },
});

// Add GPU chart for constant GPU usage
let gpuChart = new Chart(ctxGpu, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "GPU Usage (%)",
        data: [],
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  },
  options: { scales: { y: { beginAtZero: true, max: 100 } } },
});

function updateCharts(data) {
  const time = new Date().toLocaleTimeString();

  // Add the time label and data points for each chart
  cpuChart.data.labels.push(time);
  cpuChart.data.datasets[0].data.push(data.cpu);
  cpuChart.update();

  memoryChart.data.labels.push(time);
  memoryChart.data.datasets[0].data.push(data.memory);
  memoryChart.update();

  // GPU usage is constant at 60%
  gpuChart.data.labels.push(time);
  gpuChart.data.datasets[0].data.push(30); // Constant GPU usage
  gpuChart.update();

  diskChart.data.labels.push(time);
  diskChart.data.datasets[0].data.push(data.disk);
  diskChart.update();

  batteryChart.data.labels.push(time);
  batteryChart.data.datasets[0].data.push(data.battery);
  batteryChart.update();
}

// Fetch data from server and update charts every 2 seconds
function updateStats() {
  fetch("http://127.0.0.1:5000/stats")
    .then((response) => response.json())
    .then((data) => {
      updateCharts(data);
    })
    .catch((error) => console.error("Error fetching stats:", error));
}

setInterval(updateStats, 2000); 
