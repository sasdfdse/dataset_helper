from gui.app import App, install_sigint


def main():
    app = App()
    install_sigint(app)
    try:
        app.mainloop()
    except KeyboardInterrupt:  # second Ctrl+C during cleanup: exit without a traceback
        pass


if __name__ == "__main__":
    main()
