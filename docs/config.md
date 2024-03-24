# Config management

Our configuration management is done through dynaconf. This means there are a variety of ways to update a variable.
The most obvious is settings.toml, a sample of which is found in src. Whenever a python module loads config.py, it
will attempt to find a file called settings.toml relative to the python module entrypoint (the one with main). If not
found, it will recurse through local directories until it does.

Configs can also be done through envvars, prefixed with TRANSFERWARE. Ex: `TRANSFERWARE_TRAINING.EPOCHS`.

For reference on what can be configured, please reference the demo config in src. Generally, copying this config local
to your application script using transferwareai is the easiest way to modify it, as this sample settings is not itself
going to be loaded.