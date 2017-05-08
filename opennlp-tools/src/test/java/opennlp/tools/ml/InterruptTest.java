package opennlp.tools.ml;

import java.io.IOException;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Assert;
import org.junit.Test;

import opennlp.tools.ml.maxent.GISTrainer;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.TrainingParameters;

public class InterruptTest {

  @Test
  public void interruptGISTraining() throws Throwable{
    ExecutorService threadpool = Executors.newSingleThreadExecutor();
    HashMap<String, String> reportMap = new HashMap<>();

    // Get a GIS Trainer...
    TrainingParameters parameters = new TrainingParameters();
    parameters.put(TrainingParameters.ALGORITHM_PARAM, GISTrainer.MAXENT_VALUE);    
    parameters.put(TrainingParameters.ITERATIONS_PARAM, 100);
    parameters.put(GISTrainer.DATA_INDEXER_PARAM, GISTrainer.DATA_INDEXER_ONE_PASS_VALUE);
    EventTrainer trainer=TrainerFactory.getEventTrainer(parameters, reportMap);
    
    
    ObjectStream<Event> eventStream = PrepAttachDataUtil.createTrainingStream();
    // run the training on a separate thread
    // so we can call interrupt on the trainer...
    Future<MaxentModel> future = threadpool.submit(new TrainingThread(trainer, eventStream));

    // give the threadpool a second to get started....
    try {
      Thread.sleep(900);
    } catch (Exception e) {
      throw new RuntimeException("Caught an Interrupted exception at the wrong time!");
    }
    
    trainer.interrupt();

    try {
      future.get();
      Assert.fail("should have thrown an exception.");
    } catch (Exception e){
      // either it should throw an InteruptedException, or a CancelationException wrapped in an ExcutionException
      Assert.assertTrue(
          (e instanceof InterruptedException) || 
          ( (e instanceof ExecutionException) && (e.getCause() instanceof CancellationException)) 
       );
      
    } 
  }

  
  class TrainingThread implements Callable<MaxentModel>{
    EventTrainer trainer;
    ObjectStream<Event> eventstream;
    public TrainingThread(EventTrainer trainer, ObjectStream<Event> eventstream) {
      this.trainer = trainer;
      this.eventstream = eventstream;
    }
    @Override
    public MaxentModel call() throws Exception{
      return trainer.train(eventstream);
    }
  }
}
