# learning rate 1e-4
#nohup python -u pipeline_2dcnn.alexnet.py -D "0" -b 256 -G 0.45 -v 'conv5' -lr 0.0001 -e 50 > alexnet.conv5.1-e4.out  &
#nohup python -u pipeline_2dcnn.alexnet.py -D "2" -b 256 -G 0.45 -v 'fc6' -lr 0.0001 -e 50 > alexnet.fc6.1e-4.out &
#nohup python -u pipeline_2dcnn.alexnet.py -D "3" -b 256 -G 0.45 -v 'fc7' -lr 0.0001 -e 50 > alexnet.fc7.1e-4.out &
#nohup python -u pipeline_2dcnn.alexnet.py -D "4" -b 256 -G 0.45 -v 'fc8' -lr 0.0001 -e 50 > alexnet.fc8.1e-4.out &
# learning rate 1e-5
nohup python -u pipeline_2dcnn.alexnet.py -D "0" -b 256 -G 0.45 -v 'conv5' -lr 0.00001 -e 100 > alexnet.conv5.1e-5.action.out &
nohup python -u pipeline_2dcnn.alexnet.py -D "2" -b 256 -G 0.45 -v 'conv5' -lr 0.000001 -e 100 > alexnet.conv5.1e-6.action.out &
nohup python -u pipeline_2dcnn.alexnet.py -D "3" -b 256 -G 0.45 -v 'fc8' -lr 0.00001 -e 100 > alexnet.fc8.1e-5.action.out &
nohup python -u pipeline_2dcnn.alexnet.py -D "4" -b 256 -G 0.45 -v 'fc8' -lr 0.000001 -e 100 > alexnet.fc8.1e-6.action.out &
#nohup python -c 'print "hello"' > hello.out &
#nohup python -c 'print "foo"' > foo.out &
