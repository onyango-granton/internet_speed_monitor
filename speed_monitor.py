#!/usr/bin/env python3
"""
Internet Speed Monitor + Network Device Discovery

Monitors internet speed, discovers connected devices on the local subnet
during the wait interval, stores everything, visualizes trends, and
generates a PDF report.

Dependencies:
pip install speedtest-cli matplotlib plotly pandas reportlab

Usage:
python speed_monitor.py [--interval MINUTES] [--duration HOURS] [--live-plot]

Author: AI Assistant
"""

import time
import sqlite3
import csv
import argparse
import threading
import signal
import sys
import socket
import subprocess
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Use a GUI backend (TkAgg). Make sure python3-tk is installed on Linux.
import matplotlib
matplotlib.use("TkAgg")

import speedtest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


class SpeedTestError(Exception):
    """Custom exception for speed test errors"""
    pass


class InternetSpeedMonitor:
    """
    Internet speed + local network device discovery.
    Tracks download/upload/ping and number of connected devices; stores to SQLite/CSV;
    creates Matplotlib and Plotly plots; generates a PDF report.
    """

    def __init__(self, db_path: str = "speed_data.db", csv_path: str = "speed_data.csv"):
        self.db_path = db_path
        self.csv_path = csv_path
        self.running = False
        self.data_lock = threading.Lock()

        # Initialize database (with migration to add connected_devices if missing)
        self._init_database()

        # Data buffers for plotting
        self.timestamps: List[datetime] = []
        self.download_speeds: List[float] = []
        self.upload_speeds: List[float] = []
        self.ping_values: List[float] = []
        self.device_counts: List[int] = []

        # Stats
        self.stats = {
            'total_tests': 0,
            'failed_tests': 0,
            'avg_download': 0.0,
            'avg_upload': 0.0,
            'avg_ping': 0.0,
            'avg_devices': 0.0,
            'max_download': 0.0,
            'max_upload': 0.0,
            'min_ping': float('inf'),
            'max_devices': 0,
            'min_devices': float('inf'),
            'start_time': None
        }

    # --------------------------
    # Database init & migration
    # --------------------------
    def _init_database(self) -> None:
        """Create table if not exists and ensure 'connected_devices' column exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute('''
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

            # Check for connected_devices column; add if missing
            cur.execute("PRAGMA table_info(speed_tests)")
            cols = [row[1] for row in cur.fetchall()]
            if 'connected_devices' not in cols:
                cur.execute("ALTER TABLE speed_tests ADD COLUMN connected_devices INTEGER DEFAULT 0")
                conn.commit()
                print("✓ DB migrated: added 'connected_devices' column")

            conn.close()
            print(f"✓ Database ready: {self.db_path}")
        except Exception as e:
            print(f"✗ Database initialization failed: {e}")

    # --------------------------
    # Speed test
    # --------------------------
    def test_speed(self) -> Optional[Dict]:
        """Run a single speed test and return results dict."""
        try:
            print("🌐 Running speed test...")
            st = speedtest.Speedtest()
            st.get_best_server()
            server_info = st.get_best_server()

            download_speed = st.download() / 1_000_000  # Mbps
            upload_speed = st.upload() / 1_000_000      # Mbps
            ping = st.results.ping

            result = {
                'timestamp': datetime.now().isoformat(),
                'download_speed': round(download_speed, 2),
                'upload_speed': round(upload_speed, 2),
                'ping': round(ping, 2),
                'server_name': server_info.get('sponsor', ''),
                'server_location': f"{server_info.get('name','')}, {server_info.get('country','')}"
            }

            print("✓ Speed Test Complete:")
            print(f"  Download: {result['download_speed']} Mbps")
            print(f"  Upload:   {result['upload_speed']} Mbps")
            print(f"  Ping:     {result['ping']} ms")
            print(f"  Server:   {result['server_location']}")
            return result

        except Exception as e:
            print(f"✗ Speed test failed: {e}")
            self.stats['failed_tests'] += 1
            return None

    # --------------------------
    # Network device discovery
    # --------------------------
    def _get_local_ip_and_subnet(self) -> (str, ipaddress.IPv4Network):
        """Determine local IP and /24 subnet."""
        try:
            # UDP trick to learn outbound interface IP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
        except Exception:
            local_ip = socket.gethostbyname(socket.gethostname())

        # Use a /24 by default (typical home/office)
        net = ipaddress.ip_network(local_ip + "/24", strict=False)
        return local_ip, net

    def _ping(self, ip: str, timeout_sec: float = 1.0) -> bool:
        """Ping an IP once; return True if alive."""
        # Cross-platform ping
        if sys.platform.startswith("win"):
            cmd = ["ping", "-n", "1", "-w", str(int(timeout_sec * 1000)), ip]
        else:
            cmd = ["ping", "-c", "1", "-W", str(int(timeout_sec)), ip]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0
        except Exception:
            return False

    def discover_devices(self, max_workers: int = 128, timeout_sec: float = 1.0) -> int:
        """
        Scan the local /24 subnet by pinging all hosts in parallel.
        Returns the number of responding devices (including our own).
        """
        try:
            local_ip, subnet = self._get_local_ip_and_subnet()
            print(f"🔎 Scanning subnet: {subnet} (local IP {local_ip})")
            ips = [str(h) for h in subnet.hosts()]

            alive = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self._ping, ip, timeout_sec): ip for ip in ips if ip != local_ip}
                for fut in as_completed(futures):
                    if fut.result():
                        alive += 1

            # Include our own device
            total = alive + 1
            print(f"✓ Devices found: {total}")
            return total
        except Exception as e:
            print(f"⚠ Device discovery failed: {e}")
            # At least we know our own device is present
            return 1

    # --------------------------
    # Storage
    # --------------------------
    def store_result(self, result: Dict) -> None:
        """Persist to SQLite + CSV and update in-memory arrays/stats."""
        if not result:
            return

        with self.data_lock:
            try:
                # DB
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute('''
                    INSERT INTO speed_tests 
                    (timestamp, download_speed, upload_speed, ping, server_name, server_location, connected_devices)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['timestamp'],
                    result['download_speed'],
                    result['upload_speed'],
                    result['ping'],
                    result.get('server_name', ''),
                    result.get('server_location', ''),
                    result.get('connected_devices', 0),
                ))
                conn.commit()
                conn.close()

                # CSV
                file_exists = Path(self.csv_path).exists()
                with open(self.csv_path, 'a', newline='') as f:
                    fieldnames = [
                        'timestamp', 'download_speed', 'upload_speed', 'ping',
                        'server_name', 'server_location', 'connected_devices'
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result)

                # In-memory
                ts = datetime.fromisoformat(result['timestamp'])
                self.timestamps.append(ts)
                self.download_speeds.append(result['download_speed'])
                self.upload_speeds.append(result['upload_speed'])
                self.ping_values.append(result['ping'])
                self.device_counts.append(result.get('connected_devices', 0))

                self._update_stats(result)
                print("✓ Data stored")
            except Exception as e:
                print(f"✗ Failed to store data: {e}")

    def _update_stats(self, result: Dict) -> None:
        """Update running statistics."""
        self.stats['total_tests'] += 1
        if self.stats['start_time'] is None:
            self.stats['start_time'] = result['timestamp']

        n = self.stats['total_tests']
        self.stats['avg_download'] = ((self.stats['avg_download'] * (n - 1)) + result['download_speed']) / n
        self.stats['avg_upload'] = ((self.stats['avg_upload'] * (n - 1)) + result['upload_speed']) / n
        self.stats['avg_ping'] = ((self.stats['avg_ping'] * (n - 1)) + result['ping']) / n
        devices = result.get('connected_devices', 0)
        self.stats['avg_devices'] = ((self.stats['avg_devices'] * (n - 1)) + devices) / n

        self.stats['max_download'] = max(self.stats['max_download'], result['download_speed'])
        self.stats['max_upload'] = max(self.stats['max_upload'], result['upload_speed'])
        self.stats['min_ping'] = min(self.stats['min_ping'], result['ping'])
        self.stats['max_devices'] = max(self.stats['max_devices'], devices)
        self.stats['min_devices'] = min(self.stats['min_devices'], devices)

    def load_existing_data(self) -> None:
        """Load existing rows into memory (handles legacy rows without devices)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute('SELECT timestamp, download_speed, upload_speed, ping, connected_devices FROM speed_tests ORDER BY timestamp')
            rows = cur.fetchall()
            conn.close()

            with self.data_lock:
                self.timestamps.clear()
                self.download_speeds.clear()
                self.upload_speeds.clear()
                self.ping_values.clear()
                self.device_counts.clear()

                for row in rows:
                    ts = datetime.fromisoformat(row[0])
                    self.timestamps.append(ts)
                    self.download_speeds.append(row[1])
                    self.upload_speeds.append(row[2])
                    self.ping_values.append(row[3])
                    self.device_counts.append(row[4] if row[4] is not None else 0)

            print(f"✓ Loaded {len(rows)} existing records")
        except Exception as e:
            print(f"⚠ Could not load existing data: {e}")

    # --------------------------
    # Plotting
    # --------------------------
    def create_static_plots(self, save_path: str = "speed_plots.png") -> str:
        """Create static plots; annotate markers with device counts."""
        if not self.timestamps:
            print("⚠ No data available for plotting")
            return ""

        with self.data_lock:
            ts = self.timestamps.copy()
            dl = self.download_speeds.copy()
            ul = self.upload_speeds.copy()
            pg = self.ping_values.copy()
            devs = self.device_counts.copy()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

        # Speeds
        ax1.plot(ts, dl, label='Download', marker='o', linewidth=2)
        ax1.plot(ts, ul, label='Upload', marker='s', linewidth=2)
        for i, (x, y_dl, y_ul, d) in enumerate(zip(ts, dl, ul, devs)):
            y = max(y_dl, y_ul)
            ax1.annotate(f'{d}', (x, y), xytext=(0, 8 if i % 2 == 0 else -10),
                         textcoords='offset points', ha='center', fontsize=8, alpha=0.8)
        ax1.set_ylabel('Mbps')
        ax1.set_title('Internet Speed Over Time (annotated with connected devices)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Ping
        ax2.plot(ts, pg, label='Ping (ms)', marker='^', linewidth=2)
        for i, (x, y, d) in enumerate(zip(ts, pg, devs)):
            ax2.annotate(f'{d}', (x, y), xytext=(0, 8 if i % 2 == 0 else -10),
                         textcoords='offset points', ha='center', fontsize=8, alpha=0.8)
        ax2.set_ylabel('ms')
        ax2.set_title('Ping Over Time (annotated with connected devices)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Devices
        ax3.plot(ts, devs, label='Connected Devices', marker='d', linewidth=2)
        ax3.set_ylabel('Devices')
        ax3.set_xlabel('Time')
        ax3.set_title('Connected Devices Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        for ax in (ax1, ax2, ax3):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Static plots saved: {save_path}")
        return save_path

    def create_interactive_plot(self, save_path: str = "interactive_speed_plot.html") -> str:
        """Plotly interactive visualization with device counts in hover."""
        if not self.timestamps:
            print("⚠ No data available for plotting")
            return ""

        with self.data_lock:
            ts = self.timestamps.copy()
            dl = self.download_speeds.copy()
            ul = self.upload_speeds.copy()
            pg = self.ping_values.copy()
            devs = self.device_counts.copy()

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Internet Speed Over Time', 'Ping Over Time', 'Connected Devices Over Time'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
            vertical_spacing=0.08
        )

        fig.add_trace(go.Scatter(x=ts, y=dl, name='Download',
                                 mode='lines+markers',
                                 hovertext=[f"Download: {v} Mbps<br>Devices: {d}" for v, d in zip(dl, devs)],
                                 hoverinfo='text'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=ul, name='Upload',
                                 mode='lines+markers',
                                 hovertext=[f"Upload: {v} Mbps<br>Devices: {d}" for v, d in zip(ul, devs)],
                                 hoverinfo='text'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=pg, name='Ping',
                                 mode='lines+markers',
                                 hovertext=[f"Ping: {v} ms<br>Devices: {d}" for v, d in zip(pg, devs)],
                                 hoverinfo='text'), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts, y=devs, name='Devices',
                                 mode='lines+markers'), row=3, col=1)

        fig.update_layout(title="Internet Speed + Network Devices Dashboard", height=1000, showlegend=True)
        fig.update_yaxes(title_text="Mbps", row=1, col=1)
        fig.update_yaxes(title_text="ms", row=2, col=1)
        fig.update_yaxes(title_text="Devices", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)

        fig.write_html(save_path)
        print(f"✓ Interactive plot saved: {save_path}")
        return save_path

    def start_live_plot(self) -> None:
        """Live-updating Matplotlib plot (runs in a background thread if chosen)."""
        def update_plot(_frame):
            if not self.timestamps:
                return
            with self.data_lock:
                ts = self.timestamps.copy()
                dl = self.download_speeds.copy()
                ul = self.upload_speeds.copy()
                pg = self.ping_values.copy()
                devs = self.device_counts.copy()

            ax1.clear(); ax2.clear(); ax3.clear()

            ax1.plot(ts, dl, label='Download', marker='o', linewidth=2)
            ax1.plot(ts, ul, label='Upload', marker='s', linewidth=2)
            for i, (x, y_dl, y_ul, d) in enumerate(zip(ts, dl, ul, devs)):
                ax1.annotate(f'{d}', (x, max(y_dl, y_ul)),
                             xytext=(0, 8 if i % 2 == 0 else -10),
                             textcoords='offset points', ha='center', fontsize=8, alpha=0.8)
            ax1.set_ylabel('Mbps'); ax1.set_title(f'Internet Speed (n={len(ts)})'); ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2.plot(ts, pg, label='Ping', marker='^', linewidth=2)
            for i, (x, y, d) in enumerate(zip(ts, pg, devs)):
                ax2.annotate(f'{d}', (x, y),
                             xytext=(0, 8 if i % 2 == 0 else -10),
                             textcoords='offset points', ha='center', fontsize=8, alpha=0.8)
            ax2.set_ylabel('ms'); ax2.set_title('Ping'); ax2.legend(); ax2.grid(True, alpha=0.3)

            ax3.plot(ts, devs, label='Devices', marker='d', linewidth=2)
            ax3.set_ylabel('Devices'); ax3.set_xlabel('Time'); ax3.set_title('Connected Devices'); ax3.legend(); ax3.grid(True, alpha=0.3)

            for ax in (ax1, ax2, ax3):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        plt.tight_layout()
        # Keep a reference to avoid garbage collection warnings
        self.ani = FuncAnimation(fig, update_plot, interval=10000, cache_frame_data=False)
        plt.show()

    # --------------------------
    # Reporting
    # --------------------------
    def generate_report(self, output_path: str = "speed_report.pdf") -> str:
        """Generate a PDF report including device counts."""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, spaceAfter=20, alignment=1)
            story.append(Paragraph("Internet Speed Monitor Report<br/>with Network Device Discovery", title_style))
            story.append(Spacer(1, 12))

            if self.stats['total_tests'] > 0:
                summary_data = [
                    ['Metric', 'Value'],
                    ['Total Tests', str(self.stats['total_tests'])],
                    ['Failed Tests', str(self.stats['failed_tests'])],
                    ['Success Rate', f"{((self.stats['total_tests'] - self.stats['failed_tests']) / self.stats['total_tests'] * 100):.1f}%"],
                    ['Avg Download', f"{self.stats['avg_download']:.2f} Mbps"],
                    ['Avg Upload', f"{self.stats['avg_upload']:.2f} Mbps"],
                    ['Avg Ping', f"{self.stats['avg_ping']:.2f} ms"],
                    ['Avg Devices', f"{self.stats['avg_devices']:.1f}"],
                    ['Max Download', f"{self.stats['max_download']:.2f} Mbps"],
                    ['Max Upload', f"{self.stats['max_upload']:.2f} Mbps"],
                    ['Min Ping', f"{self.stats['min_ping']:.2f} ms"],
                    ['Max Devices', f"{self.stats['max_devices']}"],
                    ['Min Devices', f"{self.stats['min_devices']}"],
                ]
                tbl = Table(summary_data)
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(Paragraph("Summary Statistics", styles['Heading2']))
                story.append(tbl)
                story.append(Spacer(1, 18))

            plot_path = self.create_static_plots("temp_plot_for_report.png")
            if plot_path and Path(plot_path).exists():
                story.append(Paragraph("Trends & Annotations", styles['Heading2']))
                story.append(Image(plot_path, width=7*inch, height=8.4*inch))
                story.append(Spacer(1, 18))

            if self.timestamps:
                story.append(Paragraph("Recent Results", styles['Heading2']))
                recent_data = [['Timestamp', 'Download (Mbps)', 'Upload (Mbps)', 'Ping (ms)', 'Devices']]
                start = max(0, len(self.timestamps) - 20)
                for i in range(start, len(self.timestamps)):
                    recent_data.append([
                        self.timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
                        f"{self.download_speeds[i]:.2f}",
                        f"{self.upload_speeds[i]:.2f}",
                        f"{self.ping_values[i]:.2f}",
                        str(self.device_counts[i]) if i < len(self.device_counts) else "0"
                    ])
                recent_tbl = Table(recent_data)
                recent_tbl.setStyle(TableStyle([
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
                story.append(recent_tbl)

            story.append(Spacer(1, 20))
            story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))

            doc.build(story)

            if Path("temp_plot_for_report.png").exists():
                Path("temp_plot_for_report.png").unlink()

            print(f"✓ Report generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"✗ Report generation failed: {e}")
            return ""

    # --------------------------
    # Orchestration
    # --------------------------
    def monitor(self, interval_minutes: int = 5, duration_hours: Optional[int] = None) -> None:
        """Run tests on a schedule; discover devices between tests."""
        self.running = True
        self.load_existing_data()

        print("🚀 Starting Internet Speed Monitor + Device Discovery")
        print(f"📊 Test interval: {interval_minutes} minutes")
        if duration_hours:
            print(f"⏱ Duration: {duration_hours} hours")
            end_time = datetime.now() + timedelta(hours=duration_hours)
        else:
            print("⏱ Duration: Indefinite (Ctrl+C to stop)")
            end_time = None
        print(f"💾 Data: {self.db_path}  |  {self.csv_path}")
        print("=" * 60)

        try:
            while self.running:
                if end_time and datetime.now() >= end_time:
                    print("\n⏰ Monitoring duration completed")
                    break

                # 1) Speed test
                result = self.test_speed()

                # 2) During the wait period, discover devices (right after the test)
                device_count = self.discover_devices()

                # 3) Save combined result
                if result:
                    result['connected_devices'] = device_count
                    self.store_result(result)

                # 4) Sleep until next run (interruptible)
                print(f"💤 Waiting {interval_minutes} minutes until next test...")
                print("=" * 60)
                for _ in range(interval_minutes * 60):
                    if not self.running:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

        self.running = False

        print("\n📊 Generating final outputs...")
        self.generate_report()
        self.create_interactive_plot()

        print("\n📈 Final Statistics:")
        print(f"  Total tests: {self.stats['total_tests']}")
        print(f"  Failed tests: {self.stats['failed_tests']}")
        if self.stats['total_tests'] > 0:
            success = (self.stats['total_tests'] - self.stats['failed_tests']) / self.stats['total_tests'] * 100
            print(f"  Success rate: {success:.1f}%")
            print(f"  Avg download: {self.stats['avg_download']:.2f} Mbps")
            print(f"  Avg upload:   {self.stats['avg_upload']:.2f} Mbps")
            print(f"  Avg ping:     {self.stats['avg_ping']:.2f} ms")
            print(f"  Avg devices:  {self.stats['avg_devices']:.1f}")
            print(f"  Devices range: {self.stats['min_devices']}-{self.stats['max_devices']}")
        print("\n✅ Monitoring session completed!")

    def stop(self) -> None:
        """Stop the monitoring process."""
        self.running = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Received interrupt signal...")
    global monitor_instance
    if monitor_instance:
        monitor_instance.stop()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Internet Speed Monitor + Network Device Discovery')
    parser.add_argument('--interval', type=int, default=5, help='Test interval in minutes (default: 5)')
    parser.add_argument('--duration', type=int, help='Monitoring duration in hours (default: indefinite)')
    parser.add_argument('--live-plot', action='store_true', help='Show live updating plots')
    parser.add_argument('--db-path', default='speed_data.db', help='Database file path (default: speed_data.db)')
    parser.add_argument('--csv-path', default='speed_data.csv', help='CSV file path (default: speed_data.csv)')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    global monitor_instance
    monitor_instance = InternetSpeedMonitor(args.db_path, args.csv_path)

    if args.live_plot:
        # Warning: GUI from background threads may print a warning; this still works on most setups.
        plot_thread = threading.Thread(target=monitor_instance.start_live_plot, daemon=True)
        plot_thread.start()
        print("📈 Live plotting started in a separate window")
        time.sleep(2)

    monitor_instance.monitor(args.interval, args.duration)


if __name__ == "__main__":
    main()
