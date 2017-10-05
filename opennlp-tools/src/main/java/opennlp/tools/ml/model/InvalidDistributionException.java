package opennlp.tools.ml.model;

public class InvalidDistributionException extends RuntimeException {

  public InvalidDistributionException() {
    super();
  }

  public InvalidDistributionException(String message, Throwable cause,
      boolean enableSuppression, boolean writableStackTrace) {
    super(message, cause, enableSuppression, writableStackTrace);
  }

  public InvalidDistributionException(String message, Throwable cause) {
    super(message, cause);
  }

  public InvalidDistributionException(String message) {
    super(message);
  }

  public InvalidDistributionException(Throwable cause) {
    super(cause);
  }

}
