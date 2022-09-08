from dicomnode.server.nodes import AbstractPipeline

class storeNode(AbstractPipeline):
  log_path = "playground/exampel/logs/log.log"
  ae_title = "EXAMPLE"
  port     = 11112

if __name__ == "__main__":
  storeNode()