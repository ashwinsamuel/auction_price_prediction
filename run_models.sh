for mul in 8 1 2 3 4 5 6 7 9 10
do
  for freq in 20 60
  do
    echo "Running mul=$mul, freq=$freq → back${mul}_${freq}.log"
    python -u backtest.py $mul $freq > back${mul}_${freq}.log 2>&1
  done
done
    