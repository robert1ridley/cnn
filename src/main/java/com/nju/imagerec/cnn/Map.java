package com.nju.imagerec.cnn;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class Map extends Mapper<LongWritable, Text, Text, DoubleWritable>{
	int count = 0;
	double[][] img_train = new double [10000][784];
	double[] train_labels = new double [10000];
	
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException, NoSuchElementException {
		
		String sample = value.toString();
		StringTokenizer stringTokenizer = new StringTokenizer(sample);
		StringBuilder X = new StringBuilder(); 
		
		Double label = 0.0;
//		double [] imageTrain = new double [784];
		
		int i = 0;
		while (stringTokenizer.hasMoreTokens()) {
			if (i == 784) {
				label = Double.parseDouble(stringTokenizer.nextToken());
			}
			else {
				String term = stringTokenizer.nextToken();
				X.append(term);
				img_train[count][i] = Double.parseDouble(term);
			}
			i++;
		}
		count ++;
		
	
		Text newkeyText = new Text (X.toString());
		DoubleWritable newVal = new DoubleWritable (label);
		
		//目前输出是数据和标签，但是我们需要输出权重！
		context.write(newkeyText, newVal);
	}
	
	public void cleanup(Context context) {
		int ii=0;
	    int n_x=784;
	    int n_h=32;
	    int n_y=10;
	    NeuralNetwork neuralNetwork = new NeuralNetwork();
	    neuralNetwork.initialize_with_zeros(n_x,n_h,n_y);
	    System.out.println(img_train.length);
		for(int i =0;i<img_train.length;i++){
	        double label_train[]=new double[10];
			for(int i1=0;i1<10;i1++){
				label_train[i1]=0;
			}
			double ttt=0.001;
	        if(i>10){  
			    ttt=ttt*0.999;
				}
	        label_train[(int)train_labels[i]]=1.0;
	        double[] imgvector=neuralNetwork.narmalize_data(img_train[i]);  
	        neuralNetwork.forward_propagation(imgvector);
	        double costl=neuralNetwork.costloss(label_train);  
	        neuralNetwork.back_propagation(imgvector, label_train); 
	        neuralNetwork.update_para(ttt);
	    }
		//目标是在这里把这个map算过的权重传给reduced，然后在Reducer里算所有收到的的权重的平均
//		context.write(weights);
	}
	
}
