import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xml.etree.ElementTree import parse

import aco

# Φορτοσε το αρχειο kml
tree = parse('locations.kml')
kml = tree.getroot()

placemarks = kml.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
# Open the output CSV file for writing
with open('locations.csv', mode='w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Longitude', 'Latitude'])

    # Iterate over the placemarks and extract the location information
    for placemark in placemarks:
        # Get the coordinates of the placemark
        coordinates = placemark.find(
            './/{http://www.opengis.net/kml/2.2}coordinates').text.strip()

        # Split the coordinates into longitude, latitude, and altitude
        lon, lat, _ = [float(coord) for coord in coordinates.split(',')]

        # Write the location information to the CSV file
        csv_writer.writerow([lon, lat])

# Load the location data from the CSV file
with open('locations.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row
    locations = [(float(row[0]), float(row[1])) for row in csv_reader]

# Separate the longitude and latitude values into separate lists
lons, lats = zip(*locations)
points = np.array(locations)

# Set the x and y axis limits
plt.xlim(min(lons) - 0.001, max(lons) + 0.001)
plt.ylim(min(lats) - 0.001, max(lats) + 0.001)

plt.scatter(points[:, 0], points[:, 1])

# Add axis labels and a title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Location Map')

# Show the plot
plt.show()

dist_map = aco.distance_matrix(points)
sns.heatmap(dist_map)
plt.show()

inv_dist_map = aco.inverse_distance_matrix(points)
sns.heatmap(inv_dist_map)
plt.show()

best_path, monitor_cost = aco.aco(points=points,
                                  alpha=1,
                                  beta=1,
                                  evapo_coef=0.05,
                                  colony_size=60,
                                  num_iter=5000)

print(monitor_cost[-1])
print('')

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(points[best_path, 0], points[best_path, 1])

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(monitor_cost)), monitor_cost)
plt.show()

# αποθηκευσε το καλυτερο path
with open('path.csv', mode='w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Longitude', 'Latitude'])

    for i in range(len(points)):
        # Write the location information to the CSV file
        csv_writer.writerow([points[best_path[i], 0], points[best_path[i], 1]])

