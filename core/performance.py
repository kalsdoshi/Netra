"""
Performance Monitoring and Metrics
Tracks processing speed, bottlenecks, and generates optimization reports.
"""

import time
from collections import defaultdict
from typing import Dict, List
from datetime import datetime


class PerformanceMonitor:
    """Monitor processing performance and identify bottlenecks."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
        self.active_operations = {}
    
    def start_operation(self, op_name: str):
        """Start timing an operation."""
        self.active_operations[op_name] = time.time()
    
    def end_operation(self, op_name: str, count: int = 1):
        """
        End timing an operation and record metric.
        
        Args:
            op_name: Name of the operation
            count: Number of items processed (for rate calculation)
        """
        if op_name not in self.active_operations:
            return
        
        elapsed = time.time() - self.active_operations[op_name]
        rate = count / elapsed if elapsed > 0 else 0
        
        self.metrics[op_name].append({
            'elapsed': elapsed,
            'count': count,
            'rate': rate,
            'timestamp': datetime.now()
        })
        
        del self.active_operations[op_name]
    
    def record_metric(self, op_name: str, value: float, count: int = 1):
        """Record a metric directly."""
        self.metrics[op_name].append({
            'elapsed': value,
            'count': count,
            'rate': count / value if value > 0 else 0,
            'timestamp': datetime.now()
        })
    
    def get_average_rate(self, op_name: str) -> float:
        """Get average processing rate (items/second) for an operation."""
        if op_name not in self.metrics or not self.metrics[op_name]:
            return 0
        
        total_count = sum(m['count'] for m in self.metrics[op_name])
        total_time = sum(m['elapsed'] for m in self.metrics[op_name])
        
        return total_count / total_time if total_time > 0 else 0
    
    def get_stats(self, op_name: str) -> Dict:
        """Get detailed statistics for an operation."""
        if op_name not in self.metrics or not self.metrics[op_name]:
            return {}
        
        records = self.metrics[op_name]
        rates = [m['rate'] for m in records]
        
        return {
            'count': len(records),
            'total_items': sum(m['count'] for m in records),
            'total_time': sum(m['elapsed'] for m in records),
            'avg_rate': sum(rates) / len(rates) if rates else 0,
            'min_rate': min(rates) if rates else 0,
            'max_rate': max(rates) if rates else 0,
        }
    
    def print_report(self):
        """Print performance report."""
        if not self.metrics:
            print("No metrics collected yet.")
            return
        
        print("\n" + "="*70)
        print("📊 PERFORMANCE REPORT")
        print("="*70)
        
        for op_name in sorted(self.metrics.keys()):
            stats = self.get_stats(op_name)
            if stats:
                print(f"\n{op_name}:")
                print(f"  Operations: {stats['count']}")
                print(f"  Total Items: {stats['total_items']}")
                print(f"  Total Time: {stats['total_time']:.2f}s")
                print(f"  Avg Rate: {stats['avg_rate']:.2f} items/sec")
                print(f"  Speed Range: {stats['min_rate']:.2f} - {stats['max_rate']:.2f} items/sec")
        
        print("\n" + "="*70 + "\n")
    
    def get_bottleneck_analysis(self) -> List[Dict]:
        """Identify and rank bottlenecks by time spent."""
        bottlenecks = []
        
        for op_name in self.metrics.keys():
            total_time = sum(m['elapsed'] for m in self.metrics[op_name])
            bottlenecks.append({
                'operation': op_name,
                'total_time': total_time,
                'percentage': 0  # Will be calculated relative to others
            })
        
        total = sum(b['total_time'] for b in bottlenecks)
        for b in bottlenecks:
            b['percentage'] = (b['total_time'] / total * 100) if total > 0 else 0
        
        # Sort by time (descending)
        bottlenecks.sort(key=lambda x: x['total_time'], reverse=True)
        
        return bottlenecks


class BatchProcessor:
    """Helper class for batch processing with progress tracking."""
    
    def __init__(self, items, batch_size: int):
        self.items = items
        self.batch_size = batch_size
        self.total_batches = (len(items) + batch_size - 1) // batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        for i in range(0, len(self.items), self.batch_size):
            yield self.items[i:i + self.batch_size]
    
    def get_total_batches(self) -> int:
        """Get total number of batches."""
        return self.total_batches


# Global monitor instance
_monitor = None


def get_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def reset_monitor():
    """Reset global monitor."""
    global _monitor
    _monitor = PerformanceMonitor()


if __name__ == '__main__':
    # Example usage
    monitor = PerformanceMonitor()
    
    # Simulate some operations
    monitor.start_operation('face_detection')
    time.sleep(0.1)
    monitor.end_operation('face_detection', count=50)
    
    monitor.start_operation('embedding')
    time.sleep(0.2)
    monitor.end_operation('embedding', count=50)
    
    monitor.start_operation('clustering')
    time.sleep(0.15)
    monitor.end_operation('clustering', count=1)
    
    monitor.print_report()
    
    print("Bottleneck Analysis:")
    for b in monitor.get_bottleneck_analysis():
        print(f"  {b['operation']}: {b['percentage']:.1f}% ({b['total_time']:.2f}s)")
