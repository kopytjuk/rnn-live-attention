# Recurrent Live Attention

Attention mechanism for live predictions working on previous hidden vectors of last RNN/GRU/LSTM layer.

In order to use this mechanism in production a buffer containing last $W$ input vectors (windowed approach).

This can be used in order to detect relevant input timestamps and verify them in production.
