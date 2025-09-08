#!/usr/bin/env python3
"""
Internet Speed Monitor

A comprehensive tool for monitoring internet speed, storing data, 
visualizing trends, and generating reports.

Dependencies:
pip install speedtest-cli matplotlib plotly pandas reportlab sqlite3

Usage:
python speed_monitor.py [--interval MINUTES] [--duration HOURS] [--live-plot]

Author: AI Assistant
"""

import time
import sqlite3
import csv
import json
import argparse
import threading
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")


import speedtest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

  


class SpeedTestError(Exception):
    """Custom exception for speed test errors"""
    pass


class InternetSpeedMonitor:
    """
    A comprehensive internet speed monitoring system that tracks download/upload speeds
    and ping latency, stores data persistently, and generates visualizations and reports.
    """
    
    def __init__(self, db_path: str = "speed_data.db", csv_path: str = "speed_data.csv"):
        self.db_path = db_path
        self.csv_path = csv_path
        self.running = False
        self.data_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Data for live plotting
        self.timestamps = []
        self.download_speeds = []
        self.upload_speeds = []
        self.ping_values = []
        
        # Statistics
        self.stats = {
            'total_tests': 0,
            'failed_tests': 0,
            'avg_download': 0,
            'avg_upload': 0,
            'avg_ping': 0,
            'max_download': 0,
            'max_upload': 0,
            'min_ping': float('inf'),
            'start_time': None
        }
        
    def _init_database(self) -> None:
        """Initialize SQLite database with speed test results table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS speed_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    download_speed REAL NOT NULL,
                    upload_speed REAL NOT NULL,
                    ping REAL NOT NULL,
                    server_name TEXT,
                    server_location TEXT
                )
            ''')
            conn.commit()
            conn.close()
            print(f"✓ Database initialized: {self.db_path}")
        except Exception as e:
            print(f"✗ Database initialization failed: {e}")
            
    def test_speed(self) -> Optional[Dict]:
        """
        Perform a single speed test and return results
        
        Returns:
            Dict with speed test results or None if failed
        """
        try:
            print("🌐 Running speed test...")
            st = speedtest.Speedtest()
            
            # Get best server
            st.get_best_server()
            server_info = st.get_best_server()
            
            # Perform tests
            download_speed = st.download() / 1_000_000  # Convert to Mbps
            upload_speed = st.upload() / 1_000_000      # Convert to Mbps
            ping = st.results.ping
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'download_speed': round(download_speed, 2),
                'upload_speed': round(upload_speed, 2),
                'ping': round(ping, 2),
                'server_name': server_info['sponsor'],
                'server_location': f"{server_info['name']}, {server_info['country']}"
            }
            
            print(f"✓ Speed Test Complete:")
            print(f"  Download: {result['download_speed']} Mbps")
            print(f"  Upload: {result['upload_speed']} Mbps")
            print(f"  Ping: {result['ping']} ms")
            print(f"  Server: {result['server_location']}")
            
            return result
            
        except Exception as e:
            print(f"✗ Speed test failed: {e}")
            self.stats['failed_tests'] += 1
            return None
    
    def store_result(self, result: Dict) -> None:
        """Store test result in both database and CSV file"""
        if not result:
            return
            
        with self.data_lock:
            try:
                # Store in SQLite database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO speed_tests 
                    (timestamp, download_speed, upload_speed, ping, server_name, server_location)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result['timestamp'],
                    result['download_speed'],
                    result['upload_speed'],
                    result['ping'],
                    result['server_name'],
                    result['server_location']
                ))
                conn.commit()
                conn.close()
                
                # Store in CSV file
                file_exists = Path(self.csv_path).exists()
                with open(self.csv_path, 'a', newline='') as csvfile:
                    fieldnames = ['timestamp', 'download_speed', 'upload_speed', 'ping', 'server_name', 'server_location']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result)
                
                # Update live data for plotting
                timestamp = datetime.fromisoformat(result['timestamp'])
                self.timestamps.append(timestamp)
                self.download_speeds.append(result['download_speed'])
                self.upload_speeds.append(result['upload_speed'])
                self.ping_values.append(result['ping'])
                
                # Update statistics
                self._update_stats(result)
                
                print(f"✓ Data stored successfully")
                
            except Exception as e:
                print(f"✗ Failed to store data: {e}")
    
    def _update_stats(self, result: Dict) -> None:
        """Update running statistics"""
        self.stats['total_tests'] += 1
        
        if self.stats['start_time'] is None:
            self.stats['start_time'] = result['timestamp']
        
        # Running averages
        n = self.stats['total_tests']
        self.stats['avg_download'] = ((self.stats['avg_download'] * (n-1)) + result['download_speed']) / n
        self.stats['avg_upload'] = ((self.stats['avg_upload'] * (n-1)) + result['upload_speed']) / n
        self.stats['avg_ping'] = ((self.stats['avg_ping'] * (n-1)) + result['ping']) / n
        
        # Max/Min values
        self.stats['max_download'] = max(self.stats['max_download'], result['download_speed'])
        self.stats['max_upload'] = max(self.stats['max_upload'], result['upload_speed'])
        self.stats['min_ping'] = min(self.stats['min_ping'], result['ping'])
    
    def load_existing_data(self) -> None:
        """Load existing data from database for visualization"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT timestamp, download_speed, upload_speed, ping FROM speed_tests ORDER BY timestamp')
            rows = cursor.fetchall()
            conn.close()
            
            with self.data_lock:
                self.timestamps.clear()
                self.download_speeds.clear()
                self.upload_speeds.clear()
                self.ping_values.clear()
                
                for row in rows:
                    timestamp = datetime.fromisoformat(row[0])
                    self.timestamps.append(timestamp)
                    self.download_speeds.append(row[1])
                    self.upload_speeds.append(row[2])
                    self.ping_values.append(row[3])
            
            print(f"✓ Loaded {len(rows)} existing records")
            
        except Exception as e:
            print(f"⚠ Could not load existing data: {e}")
    
    def create_static_plots(self, save_path: str = "speed_plots.png") -> str:
        """Create static plots and save to file"""
        if not self.timestamps:
            print("⚠ No data available for plotting")
            return ""
        
        with self.data_lock:
            timestamps = self.timestamps.copy()
            download_speeds = self.download_speeds.copy()
            upload_speeds = self.upload_speeds.copy()
            ping_values = self.ping_values.copy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Speed plot
        ax1.plot(timestamps, download_speeds, label='Download Speed', marker='o', linewidth=2, markersize=4)
        ax1.plot(timestamps, upload_speeds, label='Upload Speed', marker='s', linewidth=2, markersize=4)
        ax1.set_ylabel('Speed (Mbps)')
        ax1.set_title('Internet Speed Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        # Ping plot
        ax2.plot(timestamps, ping_values, label='Ping', color='red', marker='^', linewidth=2, markersize=4)
        ax2.set_ylabel('Ping (ms)')
        ax2.set_xlabel('Time')
        ax2.set_title('Ping Latency Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Static plots saved: {save_path}")
        return save_path
    
    def create_interactive_plot(self, save_path: str = "interactive_speed_plot.html") -> str:
        """Create interactive Plotly visualization"""
        if not self.timestamps:
            print("⚠ No data available for plotting")
            return ""
        
        with self.data_lock:
            timestamps = self.timestamps.copy()
            download_speeds = self.download_speeds.copy()
            upload_speeds = self.upload_speeds.copy()
            ping_values = self.ping_values.copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Internet Speed Over Time', 'Ping Latency Over Time'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
            vertical_spacing=0.1
        )
        
        # Add speed traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=download_speeds, name='Download Speed',
                      line=dict(color='blue', width=2), mode='lines+markers'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=upload_speeds, name='Upload Speed',
                      line=dict(color='green', width=2), mode='lines+markers'),
            row=1, col=1
        )
        
        # Add ping trace
        fig.add_trace(
            go.Scatter(x=timestamps, y=ping_values, name='Ping',
                      line=dict(color='red', width=2), mode='lines+markers'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Internet Speed Monitor Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.update_yaxes(title_text="Speed (Mbps)", row=1, col=1)
        fig.update_yaxes(title_text="Ping (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        fig.write_html(save_path)
        print(f"✓ Interactive plot saved: {save_path}")
        return save_path
    
    def start_live_plot(self) -> None:
        """Start live updating matplotlib plot in separate thread"""
        def update_plot(frame):
            if not self.timestamps:
                return
            
            with self.data_lock:
                timestamps = self.timestamps.copy()
                download_speeds = self.download_speeds.copy()
                upload_speeds = self.upload_speeds.copy()
                ping_values = self.ping_values.copy()
            
            # Clear and redraw
            ax1.clear()
            ax2.clear()
            
            # Speed plot
            ax1.plot(timestamps, download_speeds, label='Download Speed', marker='o', linewidth=2)
            ax1.plot(timestamps, upload_speeds, label='Upload Speed', marker='s', linewidth=2)
            ax1.set_ylabel('Speed (Mbps)')
            ax1.set_title(f'Internet Speed Over Time (Tests: {len(timestamps)})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Ping plot
            ax2.plot(timestamps, ping_values, label='Ping', color='red', marker='^', linewidth=2)
            ax2.set_ylabel('Ping (ms)')
            ax2.set_xlabel('Time')
            ax2.set_title('Ping Latency Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis if we have data
            if timestamps:
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.tight_layout()
        
        # Create animation
        self.ani = FuncAnimation(fig, update_plot, interval=5000, cache_frame_data=False)
        
        # Show plot
        plt.show()
    
    def generate_report(self, output_path: str = "speed_report.pdf") -> str:
        """Generate comprehensive PDF report"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph("Internet Speed Monitor Report", title_style))
            story.append(Spacer(1, 12))
            
            # Summary statistics
            if self.stats['total_tests'] > 0:
                summary_data = [
                    ['Metric', 'Value'],
                    ['Total Tests Performed', str(self.stats['total_tests'])],
                    ['Failed Tests', str(self.stats['failed_tests'])],
                    ['Success Rate', f"{((self.stats['total_tests'] - self.stats['failed_tests']) / self.stats['total_tests'] * 100):.1f}%"],
                    ['Average Download Speed', f"{self.stats['avg_download']:.2f} Mbps"],
                    ['Average Upload Speed', f"{self.stats['avg_upload']:.2f} Mbps"],
                    ['Average Ping', f"{self.stats['avg_ping']:.2f} ms"],
                    ['Maximum Download Speed', f"{self.stats['max_download']:.2f} Mbps"],
                    ['Maximum Upload Speed', f"{self.stats['max_upload']:.2f} Mbps"],
                    ['Minimum Ping', f"{self.stats['min_ping']:.2f} ms"],
                ]
                
                table = Table(summary_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(Paragraph("Summary Statistics", styles['Heading2']))
                story.append(table)
                story.append(Spacer(1, 20))
            
            # Create and include plots
            plot_path = self.create_static_plots("temp_plot_for_report.png")
            if plot_path and Path(plot_path).exists():
                story.append(Paragraph("Speed Trends", styles['Heading2']))
                img = Image(plot_path, width=7*inch, height=5.8*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Recent test results (last 20)
            if self.timestamps:
                story.append(Paragraph("Recent Test Results", styles['Heading2']))
                
                recent_data = [['Timestamp', 'Download (Mbps)', 'Upload (Mbps)', 'Ping (ms)']]
                
                # Get last 20 results
                start_idx = max(0, len(self.timestamps) - 20)
                for i in range(start_idx, len(self.timestamps)):
                    recent_data.append([
                        self.timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
                        f"{self.download_speeds[i]:.2f}",
                        f"{self.upload_speeds[i]:.2f}",
                        f"{self.ping_values[i]:.2f}"
                    ])
                
                recent_table = Table(recent_data)
                recent_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(recent_table)
            
            # Generation info
            story.append(Spacer(1, 30))
            story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                  styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            # Clean up temporary plot file
            if Path("temp_plot_for_report.png").exists():
                Path("temp_plot_for_report.png").unlink()
            
            print(f"✓ Report generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ Report generation failed: {e}")
            return ""
    
    def monitor(self, interval_minutes: int = 5, duration_hours: Optional[int] = None) -> None:
        """
        Start monitoring internet speed at regular intervals
        
        Args:
            interval_minutes: Time between tests in minutes
            duration_hours: Total monitoring duration in hours (None for indefinite)
        """
        self.running = True
        
        # Load existing data
        self.load_existing_data()
        
        print(f"🚀 Starting Internet Speed Monitor")
        print(f"📊 Test interval: {interval_minutes} minutes")
        if duration_hours:
            print(f"⏱ Duration: {duration_hours} hours")
            end_time = datetime.now() + timedelta(hours=duration_hours)
        else:
            print("⏱ Duration: Indefinite (Ctrl+C to stop)")
            end_time = None
        
        print(f"💾 Data will be saved to: {self.db_path} and {self.csv_path}")
        print("=" * 60)
        
        try:
            while self.running:
                if end_time and datetime.now() >= end_time:
                    print("\n⏰ Monitoring duration completed")
                    break
                
                # Run speed test
                result = self.test_speed()
                if result:
                    self.store_result(result)
                
                print(f"💤 Waiting {interval_minutes} minutes until next test...")
                print("=" * 60)
                
                # Sleep with ability to interrupt
                for _ in range(interval_minutes * 60):  # Convert to seconds
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        
        self.running = False
        
        # Generate final report
        print("\n📊 Generating final report...")
        self.generate_report()
        self.create_interactive_plot()
        
        print(f"\n📈 Final Statistics:")
        print(f"  Total tests: {self.stats['total_tests']}")
        print(f"  Failed tests: {self.stats['failed_tests']}")
        if self.stats['total_tests'] > 0:
            print(f"  Success rate: {((self.stats['total_tests'] - self.stats['failed_tests']) / self.stats['total_tests'] * 100):.1f}%")
            print(f"  Average download: {self.stats['avg_download']:.2f} Mbps")
            print(f"  Average upload: {self.stats['avg_upload']:.2f} Mbps")
            print(f"  Average ping: {self.stats['avg_ping']:.2f} ms")
        
        print("\n✅ Monitoring session completed!")
    
    def stop(self) -> None:
        """Stop the monitoring process"""
        self.running = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal...")
    global monitor_instance
    if monitor_instance:
        monitor_instance.stop()
    sys.exit(0)


def main():
    """Main function to run the Internet Speed Monitor"""
    parser = argparse.ArgumentParser(description='Internet Speed Monitor')
    parser.add_argument('--interval', type=int, default=5, 
                       help='Test interval in minutes (default: 5)')
    parser.add_argument('--duration', type=int, 
                       help='Monitoring duration in hours (default: indefinite)')
    parser.add_argument('--live-plot', action='store_true',
                       help='Show live updating plots')
    parser.add_argument('--db-path', default='speed_data.db',
                       help='Database file path (default: speed_data.db)')
    parser.add_argument('--csv-path', default='speed_data.csv',
                       help='CSV file path (default: speed_data.csv)')
    
    args = parser.parse_args()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create monitor instance
    global monitor_instance
    monitor_instance = InternetSpeedMonitor(args.db_path, args.csv_path)
    
    # Start live plotting in separate thread if requested
    if args.live_plot:
        plot_thread = threading.Thread(target=monitor_instance.start_live_plot, daemon=True)
        plot_thread.start()
        print("📈 Live plotting started in separate window")
        time.sleep(2)  # Give plot window time to open
    
    # Start monitoring
    monitor_instance.monitor(args.interval, args.duration)


if __name__ == "__main__":
    main()