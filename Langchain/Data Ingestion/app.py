import win32service
import win32serviceutil
import win32api
import win32event
import servicemanager
import socket
import time
import dataingestion

# Import ETL code
# from your_etl_module import run_etl  # Import your ETL function here

class ETLService(win32serviceutil.ServiceFramework):
    _svc_name_ = "DataIngestionService"
    _svc_display_name_ = "Data Ingestion ETL Service"
    _svc_description_ = "A service to run ETL pipeline every 5 minutes."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.is_running = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_running = False
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)

        while self.is_running:
            # Run the ETL process
            dataingestion.run_etl()

            # Sleep for 5 minutes (300 seconds) before running the ETL again
            time.sleep(300)

def service_main():
    win32serviceutil.HandleCommandLine(ETLService)

if __name__ == '__main__':
    service_main()
