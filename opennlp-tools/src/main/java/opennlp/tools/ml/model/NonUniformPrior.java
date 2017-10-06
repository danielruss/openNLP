/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package opennlp.tools.ml.model;


import java.util.Arrays;

/**
 *  Provides a user-provided Prior for all contexts.  
 */
public class NonUniformPrior implements Prior {

  private int numberOfOutcomes = -1;
  private double[] logPriorDistribution;

  /**
   * Allows the user to update the prior between events.  This method assumes the 
   * user passed in a normalized distribution, this avoids the normalization process.
   * @param prior
   * @param delta
   * @throws InvalidDistributionException
   */
  public void setNormalizedPrior(double[] prior, double delta) throws InvalidDistributionException {
    this.logPriorDistribution  = new double[prior.length];
    int norm = 0;
    for (int i = 0;i < prior.length;i++) {
      if (prior[i] > (1 + delta) || prior[i] < 0) {
        throw new InvalidDistributionException("index " + i + " of the prior distribution is invalid: \n" +
            Arrays.toString(prior));
      }
      norm += prior[i];
      if (Math.abs(1 - norm) > delta) {
        throw new InvalidDistributionException("prior distribution is not normalized (sum=" + norm + ")\n" + 
            Arrays.toString(prior));
      }
      logPriorDistribution[i] = Math.log(prior[i]);;
    }
  }

  /**
   * Allows the user to update the prior between events. 
   * 
   * @param prior
   * @param delta
   * @throws InvalidDistributionException
   */
  public void setPrior(double[] prior, double delta) throws InvalidDistributionException{
    if (prior == null && numberOfOutcomes>0) setUniformPrior();
    
    // Don't use the array passed in..  The user may accidently change it!
    double[] normalizedPrior = Arrays.copyOf(prior, prior.length);
    
    // normalize the distribution
    double sum = 0;
    for (double d:prior) sum += d;
    if (sum == 0) setUniformPrior();
    
    if (Math.abs(1.0 - sum) < delta) {
      for (int i = 0;i < normalizedPrior.length; i++) {
        normalizedPrior[i] /= sum;
      }
    }
    setNormalizedPrior(normalizedPrior, delta);
  }

  public void setUniformPrior() throws InvalidDistributionException{
    if (numberOfOutcomes<=0) {
      throw new InvalidDistributionException("setLabels() not called.  The number of outcomes is unknown.");
    }
    this.logPriorDistribution = new double[numberOfOutcomes];
    for (int i = 0;i < numberOfOutcomes; i++) {
      logPriorDistribution[i] = Math.log(1. / numberOfOutcomes);
    }
  }
  
  /**
   * Reset the distribution to the log prior.
   * 
   * @param dist
   */
  public void logPrior(double[] dist) throws InvalidDistributionException {
    if (dist.length != logPriorDistribution.length) {
      throw new InvalidDistributionException(
          "prior distribution does not have the expected number of outcomes. Prior: " +
          logPriorDistribution.length + " expected: " + dist.length);
    }
    for (int i = 0 ; i < dist.length ; i++) {
      dist[i] = logPriorDistribution[i];
    }
  }

  @Override
  public void logPrior(double[] dist, int[] context) throws InvalidDistributionException {
    logPrior(dist);
  }



  @Override
  public void logPrior(double[] dist, int[] context, float[] values) throws InvalidDistributionException {
    logPrior(dist);
  }

  @Override
  public void logPrior(double[] dist, Context[] context, float[] values) throws InvalidDistributionException {
    logPrior(dist);
  }

  /**
   * In the UniformPrior, this method only sets the total number of outcomes.
   * For this prior, the information is set in the constructor.  This method
   * is an empty placeholder.
   */
  @Override
  public void setLabels(String[] outcomeLabels, String[] contextLabels) {
    this.numberOfOutcomes = outcomeLabels.length;
  }
}
