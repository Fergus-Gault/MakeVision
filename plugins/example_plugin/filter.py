from makevision.core import Filter
from collections import defaultdict


class ExampleFilter(Filter):
    def __init__(self, model):
        self.model = model

    def apply(self, results):
        """Apply the filter to the results."""
        filtered_results = []
        counts = defaultdict(int)

        if results is None or len(results) == 0 or results[0].boxes is None:
            return results

        # Sort results by confidence (highest confidence first)
        results = sorted(results, key=lambda result: max(result.boxes.conf) if hasattr(
            result.boxes, 'conf') and len(result.boxes.conf) > 0 else 0, reverse=True)

        class_limits = {
            "white": 1,
            "black": 1,
            "red": 7,
            "yellow": 7,
            "arm": 3,
            "hole": 6,
        }

        for result in results:
            classname = self.model.labels[int(result.boxes.cls[0])]
            if classname in class_limits:
                counts[classname] += 1
                if counts[classname] <= class_limits[classname]:
                    filtered_results.append(result)
            else:
                filtered_results.append(result)

        return filtered_results
