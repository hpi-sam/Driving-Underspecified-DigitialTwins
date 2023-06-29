sbatch  --gres gpu:1 --exclude=ac922-[01-02] -A Friedrich -t 72:00:00 -p sorcery --mem=40G train_encoder.sh

sbatch  --gres gpu:1 --exclude=ac922-[01-02] -A Friedrich -t 72:00:00 -p sorcery --mem=40G --dependency=after:484120 train_dynamics_engine.sh

sbatch  --gres gpu:1 --exclude=ac922-[01-02] -A Friedrich -t 72:00:00 -p sorcery --mem=40G --dependency=after:484123 train_TCP.sh