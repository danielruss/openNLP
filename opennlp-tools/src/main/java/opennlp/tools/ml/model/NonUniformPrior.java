package opennlp.tools.ml.model;

import java.util.Arrays;

/**
 *  Provides a user-provided Prior for all contexts.  
 */
public class NonUniformPrior implements Prior{

  double[] logPriorDistribution;

  public NonUniformPrior(double[] prior) throws InvalidDistributionException{
    this.logPriorDistribution  = new double[prior.length];
    int norm=0;
    for (int i=0;i<prior.length;i++) {
      if (prior[i]>1.000001 || prior[i]<0) {
        throw new InvalidDistributionException("index "+i+" of the prior distribution is invalid: \n"+Arrays.toString(prior));
      }
      norm+=prior[i];
      if (norm>1.000001) {
        throw new InvalidDistributionException("prior distribution is not normalized (sum="+norm+")\n"+Arrays.toString(prior));
      }
      logPriorDistribution[i] = Math.log(i);
    }
  }

  /**
   * Reset the distribution to the log prior.
   * 
   * @param dist
   */
  public void logPrior(double[] dist) throws InvalidDistributionException{
    if (dist.length != logPriorDistribution.length) {
      throw new InvalidDistributionException("prior distribution does not have the expected number of outcomes. Prior: " +
          logPriorDistribution.length + " expected: "+dist.length);
    }
    for (int i=0;i<dist.length;i++) {
      dist[i] = logPriorDistribution[i];
    }
  }
  
  @Override
  public void logPrior(double[] dist, int[] context) throws InvalidDistributionException{
    logPrior(dist);
  }
  
  
  
  @Override
  public void logPrior(double[] dist, int[] context, float[] values) throws InvalidDistributionException{
    logPrior(dist);
  }

  @Override
  public void logPrior(double[] dist, Context[] context, float[] values) throws InvalidDistributionException{
    logPrior(dist);
  }

  /**
   * In the UniformPrior, this method only sets the total number of outcomes.
   * For this prior, the information is set in the constructor.  This method
   * is an empty placeholder.
   */
  @Override
  public void setLabels(String[] outcomeLabels, String[] contextLabels) {
  }
}
