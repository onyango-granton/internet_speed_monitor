# 🌐 Internet Speed Monitor

A comprehensive Python-based internet speed monitoring tool that tracks download/upload speeds and ping latency, stores data persistently, generates beautiful visualizations, and creates detailed reports.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Command Line Options](#-command-line-options)
- [Output Files](#-output-files)
- [Screenshots](#-screenshots)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Functionality
- **📊 Automated Speed Testing**: Monitors download/upload speeds and ping at customizable intervals
- **💾 Dual Data Storage**: Saves results to both SQLite database and CSV files
- **📈 Multiple Visualizations**: Static charts, interactive dashboards, and live-updating plots
- **📄 Professional Reports**: Generates comprehensive PDF reports with statistics and charts
- **🔄 Real-time Updates**: Live statistics and optional real-time plotting
- **🛡️ Error Handling**: Robust error handling for network failures and interruptions

### Advanced Features
- **📱 Live Dashboard**: Real-time updating matplotlib charts while monitoring
- **🌍 Interactive Plots**: Beautiful Plotly-based interactive HTML dashboards  
- **📊 Comprehensive Statistics**: Running averages, min/max values, success rates
- **⚡ Resume Capability**: Loads existing data when restarting monitoring
- **🎯 Flexible Configuration**: Customizable intervals, durations, and file paths
- **🔧 Modular Design**: Clean, maintainable code with proper error handling

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- Internet connection for speed testing
- Python Tinker Installed

### Installation and Running

```bash
git clone https://github.com/onyango-granton/internet_speed_monitor.git
sudo apt install python3-tk
cd internet_speed_monitor
python3 speed_monitor.py
```

## 🚀 Quick Start

### Basic Monitoring (Default: 5-minute intervals)

```bash
python speed_monitor.py
```

### Monitor for 8 Hours with Live Plots

```bash
python speed_monitor.py --duration 8 --live-plot
```

### Custom Interval (Every 10 minutes)

```bash
python speed_monitor.py --interval 10
```

## 📖 Usage

### Starting the Monitor

1. **Open Terminal/Command Prompt**
2. **Navigate to script directory**
3. **Run the script** with desired options
4. **Press Ctrl+C** to stop monitoring gracefully

### Example Output

```
🚀 Starting Internet Speed Monitor
📊 Test interval: 5 minutes
⏱ Duration: Indefinite (Ctrl+C to stop)
💾 Data will be saved to: speed_data.db and speed_data.csv
============================================================
🌐 Running speed test...
✓ Speed Test Complete:
  Download: 85.42 Mbps
  Upload: 23.17 Mbps
  Ping: 12.3 ms
  Server: Example ISP, New York, US
✓ Data stored successfully
💤 Waiting 5 minutes until next test...
============================================================
```

## ⚙️ Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--interval` | `-i` | Test interval in minutes | `5` |
| `--duration` | `-d` | Total monitoring duration in hours | Indefinite |
| `--live-plot` | `-l` | Show live updating charts | Disabled |
| `--db-path` | | SQLite database file path | `speed_data.db` |
| `--csv-path` | | CSV output file path | `speed_data.csv` |
| `--help` | `-h` | Show help message | |

### Usage Examples

```bash
# Monitor for 24 hours, testing every 15 minutes
python speed_monitor.py --interval 15 --duration 24

# Enable live plotting with custom file paths
python speed_monitor.py --live-plot --db-path /path/to/data.db --csv-path /path/to/data.csv

# Quick 2-hour monitoring session
python speed_monitor.py --duration 2 --interval 3
```

## 📁 Output Files

The program generates several output files:

| File | Description | Format |
|------|-------------|---------|
| `speed_data.db` | SQLite database with all test results | Binary |
| `speed_data.csv` | CSV export of all test results | Text |
| `speed_plots.png` | Static charts showing speed trends | PNG Image |
| `interactive_speed_plot.html` | Interactive dashboard | HTML |
| `speed_report.pdf` | Comprehensive monitoring report | PDF |

### Database Schema

```sql
CREATE TABLE speed_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    download_speed REAL NOT NULL,
    upload_speed REAL NOT NULL,
    ping REAL NOT NULL,
    server_name TEXT,
    server_location TEXT
);
```

## 📸 Screenshots

### Live Plotting Window
The live plotting feature shows real-time charts that update as new speed tests complete:
- Upper chart: Download and Upload speeds over time
- Lower chart: Ping latency over time

### PDF Report Sample
The generated PDF report includes:
- Summary statistics table
- Speed trend charts
- Recent test results
- Generation timestamp

### Interactive Dashboard
The HTML dashboard provides:
- Zoomable and pannable charts
- Hover tooltips with exact values
- Professional styling
- Responsive design

## ⚙️ Configuration

### Customizing Test Intervals

Speed tests can be resource-intensive. Consider these guidelines:

- **High-frequency monitoring** (1-5 minutes): For troubleshooting or detailed analysis
- **Regular monitoring** (5-15 minutes): For daily usage monitoring  
- **Background monitoring** (15-60 minutes): For long-term trend analysis

### File Locations

By default, files are created in the script's directory:

```
project/
├── speed_monitor.py
├── speed_data.db          # SQLite database
├── speed_data.csv         # CSV export
├── speed_plots.png        # Static charts
├── interactive_speed_plot.html  # Interactive dashboard
└── speed_report.pdf       # Final report
```

### Server Selection

The script automatically selects the best speed test server based on ping. You can modify the `test_speed()` method to use specific servers if needed.

## 🔍 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'speedtest'"

**Solution**: Install the speedtest-cli package
```bash
pip install speedtest-cli
```

#### Speed tests are failing consistently

**Possible causes**:
- Unstable internet connection
- Firewall blocking speedtest
- Speed test servers are down

**Solution**: Check your internet connection and try running a manual speed test:
```bash
speedtest-cli
```

#### Permission errors when saving files

**Solution**: Ensure you have write permissions in the directory, or specify different paths:
```bash
python speed_monitor.py --db-path ~/Documents/speed_data.db --csv-path ~/Documents/speed_data.csv
```

#### Live plotting window not appearing

**Possible causes**:
- Missing display (headless environment)
- Matplotlib backend issues

**Solution**: 
- For headless systems, remove the `--live-plot` option
- Try setting the matplotlib backend:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Performance Considerations

- Speed tests consume bandwidth (typically 100-500 MB per test)
- Frequent testing may affect your internet usage
- Consider running during off-peak hours for baseline measurements

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

### Reporting Issues
- Use the issue tracker for bugs and feature requests
- Provide detailed information about your environment
- Include error messages and logs when possible

### Feature Requests
Some ideas for future enhancements:
- Web-based dashboard
- Email notifications for speed drops
- Integration with other monitoring tools
- Mobile app companion
- Network quality scoring

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **speedtest-cli**: For providing the core speed testing functionality
- **matplotlib**: For static plotting capabilities  
- **plotly**: For interactive visualizations
- **reportlab**: For PDF report generation

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing issues in the repository
3. Create a new issue with detailed information

## 🔄 Version History

- **v1.0**: Initial release with core monitoring functionality
- **v1.1**: Added live plotting and interactive dashboards
- **v1.2**: Enhanced PDF reports and error handling
- **v1.3**: Improved command-line interface and configuration options

---

**Happy Monitoring!** 🚀📊

For the latest updates and documentation, visit: [Project Repository](https://github.com/your-username/internet-speed-monitor)