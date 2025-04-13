// static/script.js

// --- District Coordinate Mapping ---
// Approximate coordinates for district centers in Punjab
const districtCoordinates = {
    "Amritsar": { lat: 31.6340, lon: 74.8723 },
    "Barnala": { lat: 30.3781, lon: 75.5466 },
    "Bathinda": { lat: 30.2110, lon: 74.9455 },
    "Faridkot": { lat: 30.6763, lon: 74.7547 },
    "Fatehgarh Sahib": { lat: 30.6442, lon: 76.3900 },
    "Fazilka": { lat: 30.4036, lon: 74.0259 },
    "Firozpur": { lat: 30.9214, lon: 74.6041 },
    "Gurdaspur": { lat: 32.0397, lon: 75.4048 },
    "Hoshiarpur": { lat: 31.5315, lon: 75.9066 },
    "Jalandhar": { lat: 31.3260, lon: 75.5762 },
    "Kapurthala": { lat: 31.3786, lon: 75.3848 },
    "Ludhiana": { lat: 30.9010, lon: 75.8573 },
    "Malerkotla": { lat: 30.5261, lon: 75.8845 },
    "Mansa": { lat: 29.9844, lon: 75.3904 },
    "Moga": { lat: 30.8042, lon: 75.1719 },
    "Pathankot": { lat: 32.2686, lon: 75.6494 },
    "Patiala": { lat: 30.3398, lon: 76.3869 },
    "Rupnagar": { lat: 30.9660, lon: 76.5279 }, // (Ropar)
    "Sahibzada Ajit Singh Nagar": { lat: 30.7046, lon: 76.7179 }, // (Mohali)
    "Sangrur": { lat: 30.2513, lon: 75.8401 },
    "Shahid Bhagat Singh Nagar": { lat: 31.0977, lon: 76.1079 }, // (Nawanshahr)
    "Sri Muktsar Sahib": { lat: 30.4761, lon: 74.5128 },
    "Tarn Taran": { lat: 31.4524, lon: 74.9237 }
};

// --- Populate District Dropdown ---
function populateDistrictDropdown() {
    const districtSelect = document.getElementById('district');
    const districts = Object.keys(districtCoordinates).sort(); // Get names and sort alphabetically
    districts.forEach(districtName => {
        const option = document.createElement('option');
        option.value = districtName;
        option.textContent = districtName;
        districtSelect.appendChild(option);
    });
}


// --- Map Initialization ---
let map = null;
let markersLayer = null;

function initMap() {
    if (map) { map.remove(); }
    map = L.map('map').setView([31.1471, 75.3412], 7); // Center on Punjab
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    markersLayer = L.layerGroup().addTo(map);
}

// --- Run Initialization Tasks ---
document.addEventListener('DOMContentLoaded', () => {
    populateDistrictDropdown(); // Populate districts first
    initMap(); // Then initialize map
});


// --- Form Submission Logic ---
document.getElementById('match-form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Get selected values from dropdowns
    const selectedDistrict = document.getElementById('district').value;
    const selectedBloodType = document.getElementById('blood_type').value;
    const units = document.getElementById('units').value;

    // Get DOM elements
    const statusMessage = document.getElementById('status-message');
    const resultsTable = document.getElementById('results-table');
    const resultsBody = resultsTable.getElementsByTagName('tbody')[0];
    const errorMessage = document.getElementById('error-message');

    // Reset UI
    resultsBody.innerHTML = '';
    errorMessage.textContent = '';
    resultsTable.style.display = 'none';
    statusMessage.textContent = 'Searching for donors...';
    statusMessage.style.display = 'block';
    if (markersLayer) { markersLayer.clearLayers(); }

    // --- Get Coordinates from selected district ---
    if (!selectedDistrict || !districtCoordinates[selectedDistrict]) {
        errorMessage.textContent = "Please select a valid district.";
        statusMessage.style.display = 'none';
        return; // Stop if no valid district selected
    }
    const coords = districtCoordinates[selectedDistrict];
    const latitude = coords.lat;
    const longitude = coords.lon;

    // Prepare data for API
    const requestData = {
        latitude: latitude,
        longitude: longitude,
        required_blood_type: selectedBloodType,
        required_units: parseInt(units)
    };

    // --- API Call ---
    fetch('/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify(requestData),
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || `HTTP error! Status: ${response.status}`) });
        }
        return response.json();
    })
    .then(matches => {
        statusMessage.style.display = 'none';

        if (!map || !markersLayer) { /* Error check */ return; }

        // Add marker for the bank's district center
        const bankLatLng = [requestData.latitude, requestData.longitude];
        L.marker(bankLatLng, { /* Use custom red icon */
            icon: L.icon({
                iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x-red.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
             })
        })
        .addTo(markersLayer)
        .bindPopup(`<b>Approx. Bank Location (${selectedDistrict})</b><br>Need: ${requestData.required_blood_type} (${requestData.required_units} units)`);

        // Process matches
        if (matches && matches.length > 0) {
            resultsTable.style.display = 'table';
            let donorMarkers = [];

            matches.forEach(donor => {
                // Add row to table
                // Inside matches.forEach(donor => { ... });
                let row = resultsBody.insertRow();
                row.insertCell(0).textContent = donor.donor_id || 'N/A';
                row.insertCell(1).textContent = donor.blood_type || 'N/A';
                row.insertCell(2).textContent = donor.distance_km ? donor.distance_km.toFixed(2) : 'N/A';
                row.insertCell(3).textContent = donor.latitude ? donor.latitude.toFixed(4) : 'N/A';
                row.insertCell(4).textContent = donor.longitude ? donor.longitude.toFixed(4) : 'N/A';
                row.insertCell(5).textContent = donor.cluster !== undefined ? donor.cluster : 'N/A'; // ADD BACK, adjust index if needed

                // Add donor marker to map
                if (donor.latitude && donor.longitude) {
                    const donorLatLng = [donor.latitude, donor.longitude];
                    donorMarkers.push(donorLatLng);
                    L.marker(donorLatLng)
                     .addTo(markersLayer)
                     .bindPopup(`<b>Donor:</b> ${donor.donor_id || 'N/A'}<br><b>Type:</b> ${donor.blood_type || 'N/A'}<br><b>Distance:</b> ${donor.distance_km ? donor.distance_km.toFixed(2) + ' km' : 'N/A'}`);
                }
            });

            // Fit map bounds
             if (donorMarkers.length > 0) {
                let bounds = L.latLngBounds(donorMarkers).extend(bankLatLng);
                map.fitBounds(bounds, { padding: [50, 50] });
             } else { map.setView(bankLatLng, 13); }

        } else {
            statusMessage.textContent = 'No suitable donors found matching the criteria.';
            statusMessage.style.display = 'block';
            map.setView(bankLatLng, 10); // Zoom out slightly if no donors
        }
    })
    .catch(error => {
        console.error('Error fetching matches:', error);
        statusMessage.style.display = 'none';
        errorMessage.textContent = `Error finding matches: ${error.message}`;
    });
});