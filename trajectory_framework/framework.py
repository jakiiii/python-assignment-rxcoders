import numpy as np
import matplotlib.pyplot as plt


class VehicleData:
    def __init__(self, data_file):
        self.data = np.load(data_file)
        self.ids = np.unique(self.data[:, 1])

    def by_id(self, id):
        """Segment the data based on a specific ID."""
        segment = self.data[self.data[:, 1] == id]
        return segment

    def filter(self, filter_func=None):
        """Filter the data based on an arbitrary filter function."""
        if filter_func is None:
            return [self.by_id(id) for id in self.ids]
        else:
            filtered_segments = [segment for segment in self.filter_segments(filter_func)]
            return filtered_segments

    def filter_segments(self, filter_func):
        """Helper function to filter segments based on the filter function."""
        for id in self.ids:
            segment = self.by_id(id)
            if filter_func(segment):
                yield segment

    def plot(self, segments=None):
        """Plot trajectories for the given segments."""
        if segments is None:
            segments = self.filter()

        for segment in segments:
            x_values = segment[:, 2]
            y_values = segment[:, 3]
            plt.plot(x_values, y_values, label=f"ID {int(segment[0, 1])}")

        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('Trajectory Plot')
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    data_file = 'data.npy'
    obj = VehicleData(data_file)

    # Segment example
    single_id = obj.by_id(0)
    print("Segment for ID 0:")
    print(single_id)


    # Filter example
    def length(trajectory):
        return len(trajectory)


    filtered_segments = obj.filter(length)
    print("\nFiltered Segments (Based on Length):")
    for segment in filtered_segments:
        print(segment.shape[0])

    # Plot example
    obj.plot(filtered_segments)
